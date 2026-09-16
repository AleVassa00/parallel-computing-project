#!/bin/sh

set -e
cd "$(dirname "$0")"

case "${1:-build}" in
  build)     latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex ;;
  watch)     latexmk -pdf -pvc -interaction=nonstopmode main.tex ;;
  clean)     latexmk -c ;;
  distclean) latexmk -C ;;
  *) echo "uso: $0 [build|watch|clean|distclean]" >&2; exit 1 ;;
esac
