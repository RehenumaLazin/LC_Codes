#!/usr/bin/env bash
set -euo pipefail

# ===== CONFIG =====
OUTDIR="./hrrr_conus_apcp_20250701_20250710"
START_DATE="2025-07-01"
END_DATE="2025-07-10"        # inclusive
AWS_BASE="https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# Create output folder
mkdir -p "${OUTDIR}"

# Check deps
command -v wget >/dev/null 2>&1 || { echo "ERROR: wget not found"; exit 1; }
# command -v wgrib2 >/dev/null 2>&1 || { echo "ERROR: wgrib2 not found"; exit 1; }

# Helper: zero-pad
zp2(){ printf "%02d" "$1"; }

# Iterate over days
current="${START_DATE}"
while :; do
  y=$(date -u -d "${current}" +%Y)
  m=$(date -u -d "${current}" +%m)
  d=$(date -u -d "${current}" +%d)

  day_base="${AWS_BASE}/hrrr.${y}${m}${d}/conus"

  echo "==== ${current} (UTC) ===="

  # For each cycle HH, download f01 (0-1h accumulation from that cycle)
  for HH in $(seq 0 23); do
    hh=$(zp2 "${HH}")
    fn="hrrr.t${hh}z.wrfsfcf01.grib2"
    url="${day_base}/${fn}"

    # The valid hour is cycle+1
    valid_ts=$(date -u -d "${current} ${hh}:00 +1 hour" +%Y%m%d_%H)
    out_grib="${OUTDIR}/HRRR_APCP_CONUS_${valid_ts}.grib2"
    tmp_grib="${OUTDIR}/__tmp_${y}${m}${d}_${hh}_f01.grib2"

    # Skip if we already have it
    if [[ -s "${out_grib}" ]]; then
      echo "  [skip] ${out_grib}"
      continue
    fi

    echo "  [wget] ${url}"
    if ! wget -q -O "${tmp_grib}" "${url}"; then
      echo "  [warn] missing ${url} (server may not have it)."
      rm -f "${tmp_grib}"
      continue
    fi

    # Keep only APCP messages (Accumulated Precipitation)
    # This preserves native packing and metadata.
    echo "  [wgrib2] extract APCP → ${out_grib}"
    if ! wgrib2 "${tmp_grib}" -match ":APCP:" -grib "${out_grib}" >/dev/null 2>&1; then
      echo "  [warn] APCP not found in ${tmp_grib}"
      rm -f "${tmp_grib}" "${out_grib}"
      continue
    fi

    rm -f "${tmp_grib}"
  done

  # Break when current day == END_DATE
  if [[ "$(date -u -d "${current}" +%Y%m%d)" -ge "$(date -u -d "${END_DATE}" +%Y%m%d)" ]]; then
    break
  fi
  current=$(date -u -d "${current} +1 day" +%F)
done

echo "Done. Files in: ${OUTDIR}"
echo "(Note: Hourly files are named by their VALID time, e.g., HRRR_APCP_CONUS_20250701_01.grib2)"
echo "      If you also need 2025-07-01 00Z, fetch 2025-06-30 23z f01."
