# build_local_aoi_cache.py — RUN ONCE TO CACHE ALL 11 EPOCHS + DSWx-HLS LOCALLY
import os
import time
import earthaccess
import h5py
import numpy as np

OUT_DIR = r"c:\Users\E\Downloads\sarhythm_data"
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_NPZ = os.path.join(OUT_DIR, "nisar_057_026_cache.npz")

auth = earthaccess.login(persist=True)
fs = earthaccess.get_fsspec_https_session()

# 1. Discover all 11 scenes of Track 057 / Frame 026 (P05023 Provisional + P05036 Validated)
BBOX = (20.50, 47.45, 21.70, 47.70)
scenes = []
for cid in ["C2854338529-ASF", "C3622241360-ASF"]:
    res = earthaccess.search_data(concept_id=cid, bounding_box=BBOX, temporal=("2026-06-17", "2026-11-14"))
    for g in res:
        ur = g["umm"]["GranuleUR"]
        if "_057_A_026_" in ur:
            d_raw = ur.split("_")[11][:8]
            d_str = f"{d_raw[:4]}-{d_raw[4:6]}-{d_raw[6:8]}"
            scenes.append((d_str, ur, g))

scenes.sort(key=lambda x: x[0])

# UTM Zone 34N (EPSG:32634) windows (meters) — all 100% inside Track 057 Frame 026 swath:
# 1. Tiszavalki Basin + Poroszlo/Tiszafured + Abadszalok permanent water (14 km x 14 km):
TISZA_WIN = (467000.0, 481000.0, 5264000.0, 5278000.0)
# 2. Hortobagy Fishponds (Kondás-tó & main fishpond complex, 6 km x 8 km):
HORTO_WIN = (507000.0, 513000.0, 5269000.0, 5277000.0)
# 3. Built-up Urban Reference (shifted 12 km west to 532–538 km E so 100% inside swath on all 11 dates):
URBAN_WIN = (532000.0, 538000.0, 5263000.0, 5268000.0)

def get_slice(coords, cmin, cmax):
    idx = np.where((coords >= cmin) & (coords <= cmax))[0]
    return slice(int(idx[0]), int(idx[-1]) + 1)

def read_scalar_str(group, key, default="N/A"):
    if key not in group:
        return default
    val = group[key][()]
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray) and val.size == 1:
        v = val.item()
        return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
    return str(val)

cache_dict = {}
dates_list = []

print(f"Caching {len(scenes)} epochs of Track 057 / Frame 026 to {CACHE_NPZ} ...", flush=True)
for idx, (d_str, ur, g) in enumerate(scenes, 1):
    t0 = time.time()
    h5_url = [u for u in g.data_links(access="external") if u.endswith(".h5")][0]
    print(f"[{idx}/{len(scenes)}] Caching {d_str} ({ur}) ...", end=" ", flush=True)

    # Use 256 KB block_size so HDF5 B-tree lookups do not over-fetch megabytes
    with fs.open(h5_url, "rb", block_size=256 * 1024) as f:
        with h5py.File(f, "r", rdcc_nbytes=32 * 1024 * 1024, rdcc_nslots=10007) as h5:
            ident = h5["/science/LSAR/identification"]
            freq_a = h5["/science/LSAR/GCOV/grids/frequencyA"]

            crid = read_scalar_str(ident, "compositeReleaseId")
            track = int(read_scalar_str(ident, "trackNumber", "57"))
            frame = int(read_scalar_str(ident, "frameNumber", "26"))
            orbit_dir = read_scalar_str(ident, "orbitPassDirection")
            look_dir = read_scalar_str(ident, "lookDirection")
            acq_start = read_scalar_str(ident, "zeroDopplerStartTime")
            is_full = read_scalar_str(ident, "isFullFrame")
            has_exc = read_scalar_str(ident, "hasInputDataException")
            is_urg = read_scalar_str(ident, "isUrgentObservation")
            diag_flag = read_scalar_str(ident, "diagnosticModeFlag")
            pols = [p.decode("utf-8") if isinstance(p, (bytes, np.bytes_)) else str(p)
                    for p in freq_a["listOfPolarizations"][:]]

            x = freq_a["xCoordinates"][:]
            y = freq_a["yCoordinates"][:]
            epsg = int(freq_a["projection"][()])

            sx_t, sy_t = get_slice(x, *TISZA_WIN[:2]), get_slice(y, *TISZA_WIN[2:])
            sx_h, sy_h = get_slice(x, *HORTO_WIN[:2]), get_slice(y, *HORTO_WIN[2:])
            sx_u, sy_u = get_slice(x, *URBAN_WIN[:2]), get_slice(y, *URBAN_WIN[2:])

            if idx == 1:
                cache_dict["tisza_x"] = x[sx_t]
                cache_dict["tisza_y"] = y[sy_t]
                cache_dict["horto_x"] = x[sx_h]
                cache_dict["horto_y"] = y[sy_h]
                cache_dict["urban_x"] = x[sx_u]
                cache_dict["urban_y"] = y[sy_u]
                cache_dict["epsg"] = np.array(epsg)

            key_p = d_str.replace("-", "")
            cache_dict[f"{key_p}_meta"] = np.array([
                d_str, ur, crid, str(track), str(frame), orbit_dir, look_dir,
                acq_start, "+".join(pols), is_full, has_exc, is_urg, diag_flag
            ], dtype="U120")

            # Read Tiszavalki + Abadszalok window
            cache_dict[f"{key_p}_tisza_hh"] = freq_a["HHHH"][sy_t, sx_t].astype(np.float32)
            cache_dict[f"{key_p}_tisza_mask"] = freq_a["mask"][sy_t, sx_t].astype(np.uint8)
            cache_dict[f"{key_p}_tisza_looks"] = freq_a["numberOfLooks"][sy_t, sx_t].astype(np.float32)
            if "HV" in pols and "HVHV" in freq_a:
                cache_dict[f"{key_p}_tisza_hv"] = freq_a["HVHV"][sy_t, sx_t].astype(np.float32)

            # Read Hortobagy window
            cache_dict[f"{key_p}_horto_hh"] = freq_a["HHHH"][sy_h, sx_h].astype(np.float32)
            cache_dict[f"{key_p}_horto_mask"] = freq_a["mask"][sy_h, sx_h].astype(np.uint8)
            cache_dict[f"{key_p}_horto_looks"] = freq_a["numberOfLooks"][sy_h, sx_h].astype(np.float32)
            if "HV" in pols and "HVHV" in freq_a:
                cache_dict[f"{key_p}_horto_hv"] = freq_a["HVHV"][sy_h, sx_h].astype(np.float32)

            # Read Urban reference window
            cache_dict[f"{key_p}_urban_hh"] = freq_a["HHHH"][sy_u, sx_u].astype(np.float32)
            cache_dict[f"{key_p}_urban_mask"] = freq_a["mask"][sy_u, sx_u].astype(np.uint8)

            dates_list.append(d_str)
            print(f"Done in {time.time() - t0:.1f}s (CRID={crid}, Pols={'+'.join(pols)})", flush=True)

cache_dict["dates"] = np.array(dates_list, dtype="U10")
np.savez_compressed(CACHE_NPZ, **cache_dict)
print(f"\nSaved complete 11-epoch local cache: {CACHE_NPZ} ({os.path.getsize(CACHE_NPZ)/(1024**2):.1f} MB)", flush=True)

# 2. Download 3 anchor OPERA_L3_DSWX-HLS_V1 (Tile T34TDT) granules for local cross-validation
dswx_dir = os.path.join(OUT_DIR, "dswx_reference")
os.makedirs(dswx_dir, exist_ok=True)
target_windows = [("2026-06-17", "2026-06-22"), ("2026-08-27", "2026-09-12"), ("2026-10-14", "2026-10-20")]
for w_start, w_end in target_windows:
    d_res = earthaccess.search_data(
        short_name="OPERA_L3_DSWX-HLS_V1",
        bounding_box=(20.60, 47.53, 20.78, 47.66),
        temporal=(w_start, w_end)
    )
    t34_granules = [r for r in d_res if "T34TDT" in r["umm"].get("GranuleUR", "")]
    if t34_granules:
        earthaccess.download(t34_granules[:1], local_path=dswx_dir)
print(f"Downloaded OPERA DSWx-HLS reference tiles to: {dswx_dir}", flush=True)