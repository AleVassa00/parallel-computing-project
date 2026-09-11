#!/bin/sh
# Compilazione della relazione.
#
#   ./compila.sh          -> main.pdf
#   ./compila.sh watch    -> ricompila a ogni salvataggio
#   ./compila.sh clean    -> rimuove i file ausiliari
#   ./compila.sh distclean-> rimuove anche il PDF
#
# Richiede latexmk (incluso in TeX Live e MacTeX).

set -e
cd "$(dirname "$0")"

case "${1:-build}" in
  build)     latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex ;;
  watch)     latexmk -pdf -pvc -interaction=nonstopmode main.tex ;;
  clean)     latexmk -c ;;
  distclean) latexmk -C ;;
  *) echo "uso: $0 [build|watch|clean|distclean]" >&2; exit 1 ;;
esac
