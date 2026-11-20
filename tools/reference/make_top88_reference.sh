#!/usr/bin/env bash
# Regenerates tests/reference/top88/ by running the original 88-line topology optimisation code
# (Andreassen, Clausen, Schevenels, Lazarov and Sigmund, 2010) in GNU Octave.
#
# The code itself is NOT stored in this repository: its authors reserve all rights. This script
# downloads it from their site, checks that it is the file we expect, runs it and keeps only the
# numbers it prints.
#
# Needs: bash, curl, sha256sum, awk, GNU Octave. Usage: tools/reference/make_top88_reference.sh
set -euo pipefail

url="https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/top88.m"
sha="2ae6c7a9308f9da0a602d372c799cd6e43ad61c2ac034f6b54d34ead1c5477a2"

# name:arguments of top88(nelx, nely, volfrac, penal, rmin, ft). ft = 2 is the density filter, the one
# michell has; the code's other filter (ft = 1, a heuristic on the sensitivities) is not reproduced.
cases=(
  "mbb_60x20_r2.4_density:60,20,0.5,3,2.4,2"
  "mbb_150x50_r6_density:150,50,0.5,3,6,2"
)

out="$(cd "$(dirname "$0")/../.." && pwd)/tests/reference/top88"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$out" "$work/plain" "$work/full" "$work/shim"

curl -fsSL -A "Mozilla/5.0" "$url" -o "$work/plain/top88.m"
echo "$sha  $work/plain/top88.m" | sha256sum -c - >/dev/null

# The code draws the design on every pass. These empty stand-ins let it run without a screen.
for name in colormap imagesc caxis axis drawnow; do
  printf 'function varargout = %s(varargin)\nend\n' "$name" > "$work/shim/$name.m"
done

# A second copy that differs in ONE thing: the print format of the objective (all digits instead of 4
# decimals). The script checks below that both copies print the same numbers to 4 decimals.
sed 's/Obj\.:%11\.4f/Obj.: %.15g/' "$work/plain/top88.m" > "$work/full/top88.m"
[ "$(diff "$work/plain/top88.m" "$work/full/top88.m" | grep -c '^<')" = "1" ]

run() {  # run <copy> <arguments>  ->  pass,compliance,volume_fraction,max_change
  octave --no-gui --quiet --no-init-file --path "$work/shim" --path "$work/$1" --eval "top88($2)" 2>/dev/null |
    awk '/It\.:/ { print $2 "," $4 "," $6 "," $8 }'
}

for entry in "${cases[@]}"; do
  name="${entry%%:*}"
  args="${entry#*:}"
  run plain "$args" > "$work/$name.plain.csv"
  run full "$args" > "$work/$name.full.csv"
  # same number of passes, and the same objective to the 4 printed decimals
  paste -d, "$work/$name.plain.csv" "$work/$name.full.csv" |
    awk -F, '{ d = $2 - $6; if (d < 0) d = -d; if (d > 0.00006 || $1 != $5) { print "mismatch at pass " $1; exit 1 } }'
  { echo "pass,compliance,volume_fraction,max_change"; cat "$work/$name.full.csv"; } > "$out/$name.csv"
  echo "$name: $(wc -l < "$work/$name.full.csv") passes, final compliance $(tail -1 "$work/$name.full.csv" | cut -d, -f2)"
done

{
  echo "source: $url"
  echo "sha256 of top88.m: $sha"
  echo "run with: $(octave --version 2>/dev/null | head -1)"
  echo "date: $(date -u +%Y-%m-%d)"
  echo "cases (name: nelx, nely, volfrac, penal, rmin, ft):"
  printf '  %s\n' "${cases[@]}"
} > "$out/PROVENANCE.txt"
