"""Tratamento dos dados do Sensor Logger para o formato de telemetria do CARLA.

Uso, a partir da pasta src/ do repositório:
    python -m mover.tratamento.tratar_dados
    python -m mover.tratamento.tratar_dados --entrada data/csv_maua --saida data/tratado --sem-figuras

Etapas:
  1. leitura e validação dos CSVs (sensor_logger)
  2. grade uniforme de 100 Hz para o IMU + passa-baixa
  3. vetor "cima" pela gravidade e taxa de guinada
  4. defasagem do GPS (correlação entre a guinada do GPS e a do giroscópio)
  5. eixo "frente" do veículo no aparelho (ajuste contra dv/dt e v*w do GPS)
  6. fusão GPS + IMU (EKF + suavizador RTS) na grade de 20 Hz do CARLA
  7. altitude (barômetro ancorado no GPS), rampa, atitude e jerk
  8. estimativas de throttle/brake/steer e rótulo de manobra
  9. colunas do telemetria.csv do coleta_carla + colunas extras, relatório e figuras
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # execução direta: python src/mover/tratamento/tratar_dados.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from mover.config import caminho, carregar_yaml
from mover.tratamento import calibracao, fusao, geo, sensor_logger
from mover.tratamento.sinais import G0, correlacao, interpolar, moda_movel, passa_baixa

log = logging.getLogger("mover.tratamento")

# Mesma ordem do telemetria.csv gravado pelo coleta_carla.py (v7)
COLUNAS_CARLA = [
    "frame", "sim_time", "x", "y", "z", "yaw", "pitch", "roll", "speed_mps", "throttle", "brake", "steer",
    "gear", "acc_x", "acc_y", "acc_z", "gnss_lat", "gnss_lon", "gnss_alt", "imu_acc_x", "imu_acc_y",
    "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_compass", "wp_x", "wp_y", "wp_road_id",
    "wp_lane_id", "odom_m", "cloudiness", "precipitation", "sun_altitude", "n_collisions",
    "n_lane_invasions", "cmd_manobra", "cmd_throttle", "cmd_brake", "cmd_steer", "cmd_target_speed_kmh",
]

DESCRICAO_COLUNAS = {
    "frame": "índice da amostra (0..N-1), um por tick de 20 Hz",
    "sim_time": "tempo desde a primeira amostra (s)",
    "x": "x do CARLA = leste (m) a partir da referência geográfica",
    "y": "y do CARLA = -norte (m)",
    "z": "altitude relativa à primeira amostra (m); o replay usa o z da via do mapa",
    "yaw": "guinada do CARLA (graus) = rumo - 90; cresce virando à direita",
    "pitch": "arfagem (graus), nariz para cima positivo (gravidade + offset ajustado à rampa)",
    "roll": "rolagem (graus), lado direito para baixo positivo",
    "speed_mps": "velocidade fundida GPS+IMU (m/s)",
    "throttle": "cópia de throttle_est (compatibilidade com dashboard e POC)",
    "brake": "cópia de brake_est",
    "steer": "cópia de steer_est",
    "gear": "vazio (não medido)",
    "acc_x": "aceleração no mundo CARLA, eixo x (m/s², sem gravidade)",
    "acc_y": "aceleração no mundo CARLA, eixo y (m/s², sem gravidade)",
    "acc_z": "aceleração vertical (m/s², sem gravidade)",
    "gnss_lat": "latitude da trajetória fundida (graus)",
    "gnss_lon": "longitude da trajetória fundida (graus)",
    "gnss_alt": "altitude acima do nível do mar (m), barômetro ancorado no GPS",
    "imu_acc_x": "força específica no veículo, frente (m/s²; inclui a gravidade, como o IMU do CARLA)",
    "imu_acc_y": "força específica no veículo, direita (m/s²)",
    "imu_acc_z": "força específica no veículo, cima (m/s²; ~9,81 parado no plano)",
    "imu_gyro_x": "velocidade angular em x (rad/s, convenção do IMU do CARLA)",
    "imu_gyro_y": "velocidade angular em y (rad/s, convenção do IMU do CARLA)",
    "imu_gyro_z": "taxa de guinada (rad/s), positiva virando à direita; viés removido",
    "imu_compass": "rumo (rad), 0 = norte, sentido horário (como a bússola do CARLA)",
    "wp_x": "vazio: preenchido pelo replay com o waypoint do mapa",
    "wp_y": "vazio: preenchido pelo replay",
    "wp_road_id": "vazio: preenchido pelo replay",
    "wp_lane_id": "vazio: preenchido pelo replay",
    "odom_m": "distância percorrida (m)",
    "cloudiness": "vazio (clima não medido)",
    "precipitation": "vazio (clima não medido)",
    "sun_altitude": "altitude do Sol (graus) calculada pela hora e posição da gravação",
    "n_collisions": "vazio (só existe na simulação)",
    "n_lane_invasions": "vazio (só existe na simulação)",
    "cmd_manobra": "manobra rotulada (vocabulário do GeradorDadosCaminhao)",
    "cmd_throttle": "igual a throttle_est",
    "cmd_brake": "igual a brake_est",
    "cmd_steer": "igual a steer_est",
    "cmd_target_speed_kmh": "velocidade medida (km/h), usada como alvo no replay",
    "t_unix": "instante absoluto (época Unix, s)",
    "hora_local": "data e hora local da gravação (ISO 8601)",
    "rumo_graus": "rumo (graus), 0 = norte, sentido horário",
    "speed_kmh": "velocidade (km/h)",
    "acc_long": "aceleração longitudinal no veículo (m/s²), positiva acelerando",
    "acc_lat": "aceleração lateral no veículo (m/s²), positiva para a direita",
    "acc_vert": "aceleração vertical no veículo (m/s², sem gravidade)",
    "jerk_long": "variação da aceleração longitudinal (m/s³), com a aceleração suavizada em 1,5 Hz",
    "jerk_lat": "variação da aceleração lateral (m/s³), positiva para a direita",
    "yaw_rate_dps": "taxa de guinada (graus/s), positiva virando à direita",
    "curvatura": "curvatura da trajetória (1/m), positiva à direita; vazia abaixo de 1 m/s",
    "grade_pct": "rampa da via (%), positiva em subida",
    "alt_m": "altitude acima do nível do mar (m)",
    "throttle_est": "ESTIMATIVA de acelerador [0, 1] pelo modelo longitudinal do veículo do YAML",
    "brake_est": "ESTIMATIVA de freio [0, 1]",
    "steer_est": "ESTIMATIVA de volante [-1, 1] pelo modelo bicicleta; positivo à direita",
    "manobra": "rótulo: acelerar, cruzeiro, frear, curva_esq, curva_dir ou parar",
    "sigma_pos_m": "incerteza da posição fundida (m, 1 sigma por eixo)",
    "gps_hacc_m": "precisão horizontal informada pelo GPS (m), interpolada",
    "fonte": "origem da linha: real (gravação) ou sintetico (evento injetado)",
}

CASAS_DECIMAIS = {"gnss_lat": 8, "gnss_lon": 8, "t_unix": 3}


def _resolver_sinal(valor, meta: sensor_logger.Metadados, avisos: list[str]) -> tuple[float, str]:
    if valor in (None, "auto"):
        if meta.plataforma == "ios" and not meta.padronizado:
            return -1.0, "auto: iOS sem padronização"
        if meta.plataforma in ("ios", "android"):
            return 1.0, f"auto: {meta.plataforma}" + (" padronizado" if meta.padronizado else "")
        avisos.append("Plataforma desconhecida: assumindo sinal +1 (convenção Android). Confira a rampa x arfagem.")
        return 1.0, "auto: plataforma desconhecida"
    return float(valor), "forçado no YAML"


def _coluna(df: pd.DataFrame, nome: str) -> pd.Series:
    return df[nome] if nome in df.columns else pd.Series(np.nan, index=df.index)


def _preparar_fixes(loc: pd.DataFrame) -> pd.DataFrame:
    """Fixes válidos do GPS; velocidade e rumo inválidos (-1) viram NaN."""
    alt = _coluna(loc, "altitudeAboveMeanSeaLevel")
    if alt.isna().all():
        alt = _coluna(loc, "altitude")
    vel = _coluna(loc, "speed")
    vel_acc = _coluna(loc, "speedAccuracy")
    rumo = _coluna(loc, "bearing")
    rumo_acc = _coluna(loc, "bearingAccuracy")
    fixes = pd.DataFrame({
        "t": loc["t"], "t_unix": loc["t_unix"], "lat": loc["latitude"], "lon": loc["longitude"],
        "alt_msl": alt, "hacc": loc["horizontalAccuracy"],
        "speed": vel.where(vel >= 0), "speed_acc": vel_acc.where(vel_acc > 0),
        "bearing": rumo.where((rumo >= 0) & ~(rumo_acc < 0)), "bearing_acc": rumo_acc.where(rumo_acc > 0),
    })
    fixes = fixes[(fixes["hacc"] > 0) & fixes["lat"].notna() & fixes["lon"].notna()].reset_index(drop=True)
    # O iOS costuma repetir o rumo (e às vezes a velocidade) do fix anterior: valor velho.
    fixes["bearing_fresco"] = fixes["bearing"].notna() & (fixes["bearing"].diff().abs() > 1e-9)
    velha = (fixes["speed"].diff().abs() < 1e-9) & (fixes["speed"] > 0.5)
    fixes["speed_fresca"] = fixes["speed"].notna() & ~velha
    return fixes


def _referencia(cfg_ref: dict, fixes: pd.DataFrame, avisos: list[str]) -> geo.ReferenciaGeo:
    modo = cfg_ref.get("modo", "osm")
    if modo == "manual":
        return geo.ReferenciaGeo(float(cfg_ref["lat0"]), float(cfg_ref["lon0"]), "manual (YAML)")
    if modo == "osm":
        arquivo = caminho(cfg_ref["arquivo_osm"])
        if arquivo.exists():
            return geo.referencia_do_osm(arquivo)
        avisos.append(f"Arquivo OSM {arquivo} não encontrado: referência = primeiro fix do GPS.")
    return geo.ReferenciaGeo(float(fixes["lat"].iloc[0]), float(fixes["lon"].iloc[0]), "primeiro fix GPS")


def _altitude(baro: pd.DataFrame | None, fixes: pd.DataFrame, t: np.ndarray, hz: float, fc: float) -> tuple[np.ndarray, dict]:
    """Altitude acima do nível do mar: barômetro (relativo) com offset mediano contra o GPS."""
    ok = fixes["alt_msl"].notna()
    t_fix, alt_fix = fixes.loc[ok, "t_corr"].to_numpy(), fixes.loc[ok, "alt_msl"].to_numpy()
    if baro is not None and len(baro) >= 5:
        rel = passa_baixa(np.interp(t, baro["t"], baro["relativeAltitude"]), hz, fc)
        offset = float(np.median(alt_fix - np.interp(t_fix, baro["t"], baro["relativeAltitude"])))
        alt = rel + offset
        desvio = alt_fix - np.interp(t_fix, t, alt)
        return alt, {"fonte": "barômetro ancorado no GPS", "offset_m": offset,
                     "rms_vs_gps_m": float(np.sqrt(np.mean(desvio**2)))}
    alt = passa_baixa(np.interp(t, t_fix, alt_fix), hz, min(fc, 0.05))
    return alt, {"fonte": "GPS (sem barômetro)"}


def _rampa(s: np.ndarray, alt: np.ndarray, janela_m: float) -> np.ndarray:
    """dh/ds por Savitzky-Golay numa grade de distância de 1 m."""
    if s[-1] < 2 * janela_m:
        return np.zeros_like(s)
    s_u, idx = np.unique(s, return_index=True)
    grade_s = np.arange(0.0, s_u[-1], 1.0)
    h = np.interp(grade_s, s_u, alt[idx])
    dh = savgol_filter(h, int(janela_m) | 1, 2, deriv=1, delta=1.0)
    return np.interp(s, grade_s, dh)


def _estimar_controles(a_long, v, rampa, w_cima, veic: dict, vel_parado: float):
    """throttle/brake por força longitudinal necessária e steer pelo modelo bicicleta."""
    theta = np.arctan(rampa)
    a_resist = (G0 * (veic["coef_rolamento"] * np.cos(theta) + np.sin(theta))
                + 0.5 * 1.225 * veic["coef_arrasto"] * veic["area_frontal_m2"] * v**2 / veic["massa_kg"])
    a_req = a_long + a_resist
    throttle = np.clip(a_req / veic["acel_max_tracao_mps2"], 0, 1)
    brake = np.clip((-a_req - veic["freio_motor_mps2"]) / veic["desacel_max_freio_mps2"], 0, 1)
    parado = v < vel_parado
    throttle[parado], brake[parado] = 0.0, 1.0
    curv_dir = -w_cima / np.maximum(v, 1.0)
    steer = np.clip(np.arctan(veic["entre_eixos_m"] * curv_dir) / np.radians(veic["angulo_max_roda_graus"]), -1, 1)
    return throttle, brake, steer


def _rotular_manobras(a_long, w_dps, v, cfg_m: dict, hz: float) -> np.ndarray:
    """Rótulos com o vocabulário do coleta_carla v7, com filtro de moda."""
    rot = np.full(len(v), "cruzeiro", dtype=object)
    rot[a_long >= cfg_m["acel_limiar_mps2"]] = "acelerar"
    rot[a_long <= -cfg_m["acel_limiar_mps2"]] = "frear"
    rot[w_dps >= cfg_m["guinada_limiar_dps"]] = "curva_esq"
    rot[w_dps <= -cfg_m["guinada_limiar_dps"]] = "curva_dir"
    rot[v < cfg_m["vel_parado_mps"]] = "parar"
    return moda_movel(rot.astype(str), int(round(cfg_m["janela_moda_s"] * hz)) | 1)


def _hora_local(t_unix: np.ndarray, fuso: str) -> pd.Series:
    instantes = pd.Series(pd.to_datetime(t_unix, unit="s", utc=True))
    try:
        instantes = instantes.dt.tz_convert(fuso)
    except Exception:  # fuso inválido no Metadata
        pass
    deslocamento = instantes.dt.strftime("%z").str.replace(r"(\d\d)(\d\d)$", r"\1:\2", regex=True)
    return instantes.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + deslocamento


def _limpar_json(obj):
    """Converte tipos do numpy e NaN para algo serializável em JSON estrito."""
    if isinstance(obj, dict):
        return {str(k): _limpar_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_limpar_json(v) for v in (obj.tolist() if isinstance(obj, np.ndarray) else obj)]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def tratar(cfg: dict, entrada: Path, saida: Path, gerar_figuras: bool = True) -> dict:
    """Executa o tratamento completo e grava CSVs, relatório e figuras em `saida`."""
    avisos: list[str] = []
    grav = sensor_logger.carregar(entrada)
    avisos += grav.avisos
    meta = grav.meta
    sinal, origem_sinal = _resolver_sinal(cfg["sinais"].get("sinal_aceleracao", "auto"), meta, avisos)
    log.info("Plataforma %s, sinal da aceleração %+.0f (%s)", meta.plataforma, sinal, origem_sinal)

    fs, hz = float(cfg["amostragem"]["imu_hz"]), float(cfg["amostragem"]["saida_hz"])
    cf, cc, ordem = cfg["filtros"], cfg["calibracao"], cfg["filtros"]["ordem"]

    # 2. Grade uniforme do IMU
    imu = {n: grav.sensores[n] for n in ("Accelerometer", "Gravity", "Gyroscope")}
    t_imu = np.arange(max(d["t"].iloc[0] for d in imu.values()), min(d["t"].iloc[-1] for d in imu.values()), 1 / fs)
    xyz = {n: interpolar(t_imu, d["t"], d[["x", "y", "z"]].to_numpy()) for n, d in imu.items()}
    acc = passa_baixa(sinal * xyz["Accelerometer"], fs, cf["imu_corte_hz"], ordem)  # cinemática, sinal físico
    gravidade = passa_baixa(xyz["Gravity"], fs, cf["imu_corte_hz"], ordem)
    giro = passa_baixa(xyz["Gyroscope"], fs, cf["imu_corte_hz"], ordem)

    # 3. Vetor cima e guinada
    cima, cima_t, desvio = calibracao.vetor_cima(gravidade, sinal)
    w_cima = giro @ cima  # rad/s, anti-horário (esquerda) positivo
    log.info("Montagem: desvio da gravidade mediana %.1f°, máx %.1f°", desvio["mediana"], desvio["max"])

    # GPS e referência geográfica
    fixes = _preparar_fixes(grav.sensores["Location"])
    ref = _referencia(cfg["referencia_geo"], fixes, avisos)
    fixes["leste"], fixes["norte"] = ref.para_enu(fixes["lat"], fixes["lon"])
    log.info("Referência geográfica: %.7f, %.7f (%s)", ref.lat0, ref.lon0, ref.origem)

    # 4. Defasagem do GPS
    vel_min_rumo = cfg["fusao"]["vel_min_rumo_mps"]
    okv = fixes["speed_fresca"].to_numpy()
    t_vel, vel = fixes.loc[okv, "t"].to_numpy(), fixes.loc[okv, "speed"].to_numpy()
    metodo = cc.get("defasagem_metodo", "velocidade")
    if cc.get("defasagem_fixa_s") is not None:
        defas = calibracao.Defasagem(float(cc["defasagem_fixa_s"]), float("nan"), float("nan"), "fixa (YAML)")
    elif metodo == "rumo":
        defas = calibracao.estimar_defasagem_rumo(t_imu, fs, w_cima, fixes, cf, cc, vel_min_rumo)
    else:
        defas = calibracao.estimar_defasagem_velocidade(t_imu, fs, acc, w_cima, cima, t_vel, vel, cf, cc)
    if abs(defas.segundos) >= cc["defasagem_max_s"] - 1e-6:
        avisos.append(f"Defasagem do GPS no limite da busca ({defas.segundos:.2f} s): aumente defasagem_max_s.")
    fixes["t_corr"] = fixes["t"] - defas.segundos
    log.info("Defasagem do GPS (%s): %.2f s (correlação %.3f; sem correção %.3f)",
             defas.metodo, defas.segundos, defas.correlacao, defas.correlacao_sem_defasagem)

    # 5. Eixo frente e ganhos
    mont, amostras_cal = calibracao.ajustar_frente(
        t_imu, fs, acc, w_cima, cima, desvio, t_vel - defas.segundos, vel, cf, cc)
    g_long, g_lat, av = calibracao.ganhos_aplicaveis(mont, cc)
    avisos += av
    log.info("Eixo frente: R² long %.3f, R² lat %.3f, ganhos aplicados %.3f / %.3f",
             mont.r2_long, mont.r2_lat, g_long, g_lat)

    a_veic = acc @ mont.R.T  # (frente, esquerda, cima)
    a_veic[:, 0] *= g_long
    a_veic[:, 1] *= g_lat
    w_veic = giro @ mont.R.T
    cima_veic = cima_t @ mont.R.T
    f_veic = a_veic + (sinal * gravidade) @ mont.R.T  # força específica (como um acelerômetro real)

    # 6. Fusão na grade de saída
    dt = 1 / hz
    t_ini = max(t_imu[0], fixes["t_corr"].iloc[0])
    t_fim = min(t_imu[-1], fixes["t_corr"].iloc[-1] + 1.0)
    t = np.arange(math.ceil(t_ini / dt), math.floor(t_fim / dt) + 1) * dt
    fixes["sim_time"] = fixes["t_corr"] - t[0]
    cfu = cfg["fusao"]
    rumo_acc = np.radians(fixes["bearing_acc"].fillna(30.0).to_numpy())
    psi_gps = geo.rumo_para_psi(fixes["bearing"].to_numpy())
    psi_gps[fixes["speed"].fillna(0).to_numpy() < vel_min_rumo] = np.nan
    modo_rumo = cfu.get("usar_rumo_gps", "frescos")
    if modo_rumo == "frescos":
        psi_gps[~fixes["bearing_fresco"].to_numpy()] = np.nan
    elif modo_rumo == "nenhum":
        psi_gps[:] = np.nan
    vel_fusao = fixes["speed"].where(fixes["speed_fresca"]).to_numpy()
    gps_in = fusao.EntradaGPS(
        t=fixes["t_corr"].to_numpy(),
        leste=fixes["leste"].to_numpy(), norte=fixes["norte"].to_numpy(),
        sigma_pos=np.maximum(cfu["fator_sigma_pos"] * fixes["hacc"].to_numpy(), cfu["sigma_pos_min_m"]),
        vel=vel_fusao,
        sigma_vel=np.maximum(cfu["fator_sigma_vel"] * fixes["speed_acc"].fillna(1.0).to_numpy(), cfu["sigma_vel_min_mps"]),
        psi=psi_gps,
        sigma_psi=np.maximum(rumo_acc, np.radians(cfu["sigma_rumo_min_graus"])),
    )
    w20 = np.interp(t, t_imu, w_cima)
    fus = fusao.fundir(t, w20, np.interp(t, t_imu, a_veic[:, 0]), gps_in, cfu)
    fixes["residuo_m"], fixes["usado_na_fusao"] = fus.residuo_m, fus.usado
    log.info("Fusão: RMS dos resíduos de posição %.2f m (máx %.2f m), %d fixes rejeitados",
             fus.rms_residuo_m, fus.max_residuo_m, fus.n_rejeitados)

    # 7. Sinais derivados
    v = np.clip(fus.v, 0, None)
    psi = fus.psi
    vies_acel_imu = np.interp(t_imu, t, fus.vies_acel)
    w_corr = w20 - fus.vies_giro
    a_long_imu = a_veic[:, 0] - vies_acel_imu
    a_long = np.interp(t, t_imu, a_long_imu)
    a_esq = np.interp(t, t_imu, a_veic[:, 1])
    a_cima = np.interp(t, t_imu, a_veic[:, 2])
    jerk_long = np.interp(t, t_imu, np.gradient(passa_baixa(a_long_imu, fs, cf["jerk_corte_hz"], ordem), 1 / fs))
    jerk_esq = np.interp(t, t_imu, np.gradient(passa_baixa(a_veic[:, 1], fs, cf["jerk_corte_hz"], ordem), 1 / fs))

    alt, info_alt = _altitude(grav.sensores.get("Barometer"), fixes, t, hz, cf["altitude_corte_hz"])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    rampa = _rampa(s, alt, cc["rampa_janela_m"])

    arf_g, rol_g = calibracao.atitude_por_gravidade(interpolar(t, t_imu, cima_veic))
    mov = v > cc["vel_min_mps"]
    offset_arf = float(np.median(np.arctan(rampa[mov]) - arf_g[mov])) if mov.any() else 0.0
    corr_rampa = correlacao(np.arctan(rampa[mov]), arf_g[mov])
    if corr_rampa < -0.3:
        avisos.append(f"Arfagem x rampa com correlação negativa ({corr_rampa:.2f}): o sinal da aceleração "
                      "pode estar invertido. Teste sinais.sinal_aceleracao com o valor oposto.")
    arfagem = arf_g + offset_arf

    f20 = interpolar(t, t_imu, f_veic)
    f20[:, 0] -= fus.vies_acel
    w20_veic = interpolar(t, t_imu, w_veic)

    a_leste = a_long * np.cos(psi) - a_esq * np.sin(psi)
    a_norte = a_long * np.sin(psi) + a_esq * np.cos(psi)
    lat, lon = ref.de_enu(fus.leste, fus.norte)
    x_carla, y_carla = geo.enu_para_carla(fus.leste, fus.norte)
    rumo = geo.psi_para_rumo(psi)
    t_unix = meta.epoca_ns / 1e9 + t

    # 8. Estimativas e manobras
    veic, cm = cfg["veiculo"], cfg["manobras"]
    a_long_suave = passa_baixa(a_long, hz, cf["jerk_corte_hz"], ordem)
    throttle, brake, steer = _estimar_controles(a_long_suave, v, rampa, w_corr, veic, cm["vel_parado_mps"])
    manobra = _rotular_manobras(passa_baixa(a_long, hz, cf["manobra_corte_hz"], ordem),
                                np.degrees(passa_baixa(w_corr, hz, cf["manobra_corte_hz"], ordem)), v, cm, hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvatura = np.where(v >= 1.0, -w_corr / v, np.nan)

    # 9. Tabela final
    n = len(t)
    vazio = np.full(n, np.nan)
    df = pd.DataFrame({
        "frame": np.arange(n), "sim_time": t - t[0],
        "x": x_carla, "y": y_carla, "z": alt - alt[0],
        "yaw": geo.psi_para_yaw_carla(psi), "pitch": np.degrees(arfagem), "roll": np.degrees(rol_g),
        "speed_mps": v, "throttle": throttle, "brake": brake, "steer": steer, "gear": vazio,
        "acc_x": a_leste, "acc_y": -a_norte, "acc_z": a_cima,
        "gnss_lat": lat, "gnss_lon": lon, "gnss_alt": alt,
        "imu_acc_x": f20[:, 0], "imu_acc_y": -f20[:, 1], "imu_acc_z": f20[:, 2],
        "imu_gyro_x": -w20_veic[:, 0], "imu_gyro_y": w20_veic[:, 1], "imu_gyro_z": -w_corr,
        "imu_compass": np.radians(rumo),
        "wp_x": vazio, "wp_y": vazio, "wp_road_id": vazio, "wp_lane_id": vazio,
        "odom_m": s, "cloudiness": vazio, "precipitation": vazio,
        "sun_altitude": geo.altitude_solar(ref.lat0, ref.lon0, t_unix),
        "n_collisions": vazio, "n_lane_invasions": vazio,
        "cmd_manobra": manobra, "cmd_throttle": throttle, "cmd_brake": brake, "cmd_steer": steer,
        "cmd_target_speed_kmh": v * 3.6,
        "t_unix": t_unix, "hora_local": _hora_local(t_unix, meta.fuso).to_numpy(),
        "rumo_graus": rumo, "speed_kmh": v * 3.6,
        "acc_long": a_long, "acc_lat": -a_esq, "acc_vert": a_cima,
        "jerk_long": jerk_long, "jerk_lat": -jerk_esq,
        "yaw_rate_dps": -np.degrees(w_corr), "curvatura": curvatura, "grade_pct": 100 * rampa, "alt_m": alt,
        "throttle_est": throttle, "brake_est": brake, "steer_est": steer, "manobra": manobra,
        "sigma_pos_m": fus.sigma_pos,
        "gps_hacc_m": np.interp(t, fixes["t_corr"], fixes["hacc"]),
        "fonte": "real",
    })
    df = df.round({c: CASAS_DECIMAIS.get(c, 5) for c in df.select_dtypes("number").columns})

    # Saídas
    cs = cfg["saida"]
    saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / cs["telemetria"], index=False, encoding="utf-8", lineterminator="\n")
    x_fix, y_fix = geo.enu_para_carla(fixes["leste"], fixes["norte"])
    fixes.assign(x=x_fix, y=y_fix).round(8).to_csv(saida / cs["gps"], index=False, encoding="utf-8", lineterminator="\n")
    amostras_cal.round(5).to_csv(saida / cs["calibracao"], index=False, encoding="utf-8", lineterminator="\n")

    jl = df["jerk_long"].to_numpy()
    manobras_pct = (df["manobra"].value_counts(normalize=True) * 100).round(1).to_dict()
    relatorio = {
        "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "entrada": {
            "pasta": str(entrada), "dispositivo": meta.dispositivo, "plataforma": meta.plataforma,
            "padronizado": meta.padronizado, "versao_app": meta.versao_app,
            "hora_gravacao_utc_metadata": meta.hora_gravacao_utc, "inicio_local": str(df["hora_local"].iloc[0]),
            "fuso": meta.fuso, "sensores_usados": sorted(n for n in grav.sensores if n not in ("Compass", "Orientation")),
            "sensores_ignorados": {**grav.ignorados,
                                   **{n: "não usado: a bússola sofre interferência do carro e o rumo vem da fusão GPS+giroscópio"
                                      for n in ("Compass", "Orientation") if n in grav.sensores}},
        },
        "sinais": {"sinal_aceleracao": sinal, "origem": origem_sinal},
        "referencia_geo": {"lat0": ref.lat0, "lon0": ref.lon0, "origem": ref.origem, "proj_string": ref.proj_string(),
                           "convencao": "x = leste, y = -norte, yaw = rumo - 90 (graus)"},
        "montagem": {
            "cima_no_aparelho": mont.cima, "frente_no_aparelho": mont.frente, "esquerda_no_aparelho": mont.esquerda,
            "R_aparelho_para_veiculo": mont.R, "theta_graus": mont.theta_graus,
            "theta_so_longitudinal_graus": mont.theta_long_graus, "theta_so_lateral_graus": mont.theta_lat_graus,
            "r2_longitudinal": mont.r2_long, "r2_lateral": mont.r2_lat,
            "ganho_longitudinal_ajustado": mont.ganho_long, "ganho_lateral_ajustado": mont.ganho_lat,
            "ganhos_aplicados": [g_long, g_lat], "amostras": mont.n_amostras,
            "desvio_gravidade_graus": mont.desvio_gravidade_graus,
        },
        "defasagem_gps": {"segundos": defas.segundos, "metodo": defas.metodo, "correlacao": defas.correlacao,
                          "correlacao_sem_defasagem": defas.correlacao_sem_defasagem,
                          "curva": {"lags_s": defas.curva_lags, "correlacao": defas.curva_corr}},
        "fusao": {"rms_residuo_pos_m": fus.rms_residuo_m, "max_residuo_pos_m": fus.max_residuo_m,
                  "fixes_total": len(fixes), "fixes_usados": int(fus.usado.sum()), "fixes_rejeitados": fus.n_rejeitados,
                  "usar_rumo_gps": modo_rumo,
                  "rumos_repetidos_descartados": int((fixes["bearing"].notna() & ~fixes["bearing_fresco"]).sum()),
                  "velocidades_repetidas_descartadas": int((fixes["speed"].notna() & ~fixes["speed_fresca"]).sum()),
                  "vies_giro_dps": float(np.degrees(np.median(fus.vies_giro))),
                  "vies_acel_mps2": float(np.median(fus.vies_acel)),
                  "sigma_pos_mediano_m": float(np.median(fus.sigma_pos))},
        "altitude": info_alt,
        "atitude": {"offset_arfagem_graus": float(np.degrees(offset_arf)), "corr_rampa_arfagem": corr_rampa},
        "volta": {
            "amostras": n, "taxa_hz": hz, "duracao_s": float(t[-1] - t[0]), "distancia_m": float(s[-1]),
            "vel_media_kmh": float(v.mean() * 3.6), "vel_max_kmh": float(v.max() * 3.6),
            "acel_long_max_mps2": float(a_long.max()), "acel_long_min_mps2": float(a_long.min()),
            "acel_lat_max_abs_mps2": float(np.abs(a_esq).max()),
            "jerk_long_percentis_mps3": {f"p{p}": float(np.percentile(jl, p)) for p in (1, 5, 50, 95, 99)},
            "jerk_long_min_max_mps3": [float(jl.min()), float(jl.max())],
            "taxa_guinada_max_dps": float(np.degrees(np.abs(w_corr).max())),
            "guinada_total_graus": float(np.degrees(psi[-1] - psi[0])),
            "desnivel_m": float(alt.max() - alt.min()),
            "rampa_p5_p95_pct": [float(np.percentile(100 * rampa, 5)), float(np.percentile(100 * rampa, 95))],
            "distancia_inicio_fim_m": float(np.hypot(fus.leste[-1] - fus.leste[0], fus.norte[-1] - fus.norte[0])),
            "manobras_pct_tempo": manobras_pct,
        },
        "colunas": {c: DESCRICAO_COLUNAS.get(c, "") for c in df.columns},
        "avisos": avisos,
    }
    with (saida / cs["relatorio"]).open("w", encoding="utf-8") as f:
        json.dump(_limpar_json(relatorio), f, ensure_ascii=False, indent=2)

    for a in avisos:
        log.warning(a)
    log.info("Gravado %s (%d linhas x %d colunas)", saida / cs["telemetria"], n, df.shape[1])
    if gerar_figuras:
        from mover.tratamento import validar  # importa matplotlib só quando necessário

        validar.gerar_figuras(saida, cfg)
    return relatorio


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trata os CSVs do Sensor Logger para o formato do CARLA.")
    parser.add_argument("--config", default="config/tratamento.yaml", help="YAML de configuração")
    parser.add_argument("--entrada", help="pasta com os CSVs do Sensor Logger (padrão: YAML)")
    parser.add_argument("--saida", help="pasta de saída (padrão: YAML)")
    parser.add_argument("--sem-figuras", action="store_true", help="não gera as figuras de validação")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = carregar_yaml(args.config)
    entrada = caminho(args.entrada or cfg["entrada"]["pasta"])
    saida = caminho(args.saida or cfg["saida"]["pasta"])
    tratar(cfg, entrada, saida, gerar_figuras=not args.sem_figuras)


if __name__ == "__main__":
    main()
