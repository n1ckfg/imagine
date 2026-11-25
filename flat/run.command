#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

cd "$DIR"

if [ ! -f "index.html" ]; then
  echo "index.html not found next to run.command."
  exit 1
fi

PORT=8080

while lsof -iTCP:"$PORT" -sTCP:LISTEN -nP >/dev/null 2>&1; do
  PORT=$((PORT + 1))
done

if command -v http-server >/dev/null 2>&1; then
  http-server . -a 127.0.0.1 -p "$PORT" -c-1 -o
else
  npx --yes http-server . -a 127.0.0.1 -p "$PORT" -c-1 -o
fi
