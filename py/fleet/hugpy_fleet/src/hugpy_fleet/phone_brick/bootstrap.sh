#!/data/data/com.termux/files/usr/bin/bash
# hugpy phone-brick bootstrap — turn a Termux phone into a fleet worker.
#
# Central serves this templated (placeholders filled in):
#
#   curl -sL http://<central>/api/phone-brick/install.sh | bash
#   curl -sL "http://<central>/api/phone-brick/install.sh?name=note20&role=both" | bash
#
# Roles:
#   ppe   ONNX PPE detector       (python -m phone_brick worker, port 5002)
#   llm   GGUF chat-pool worker   (python -m gguf_worker,        port 9100)
#   both  both of the above
#
# Idempotent — re-run anytime; every step skips what's already done. Code is
# pulled from central's running tree (/api/phone-brick/code.tar.gz), so
# re-running is also how a phone picks up worker updates.
set -eu

CENTRAL="${PHONE_BRICK_CENTRAL:-@CENTRAL_URL@}"
NAME="${PHONE_BRICK_NAME:-@NAME@}"
ROLE="${PHONE_BRICK_ROLE:-@ROLE@}"
[ -n "$NAME" ] || NAME="$(hostname)"

case "${PREFIX:-}" in
  *com.termux*) ;;
  *) echo "ERROR: not a Termux environment (PREFIX=${PREFIX:-unset})" >&2; exit 1 ;;
esac
case "$ROLE" in ppe|llm|both) ;;
  *) echo "ERROR: role must be ppe, llm or both (got '$ROLE')" >&2; exit 1 ;;
esac

echo "==> central: $CENTRAL | name: $NAME | role: $ROLE"

# fetch <url> [outfile] — curl, falling back to python urllib. Termux partial
# upgrades can break curl mid-script (libcurl vs libngtcp2 symbol mismatch);
# python is installed before fetch is ever used, so the fallback always works.
fetch() {
  if [ -n "${2:-}" ]; then
    curl -sf "$1" -o "$2" 2>/dev/null && return 0
    python -c "import sys,urllib.request as u; u.urlretrieve(sys.argv[1], sys.argv[2])" "$1" "$2"
  else
    curl -sf "$1" 2>/dev/null && return 0
    python -c "import sys,urllib.request as u; sys.stdout.write(u.urlopen(sys.argv[1]).read().decode())" "$1"
  fi
}

echo "==> base packages (python, curl, openssh)"
yes | pkg update >/dev/null 2>&1 || true
pkg install -y python curl openssh >/dev/null
# Heal the half-upgraded state pkg install can leave curl in (missing
# ngtcp2/nghttp3 symbols in libcurl) — harmless no-op when versions align.
curl -sV >/dev/null 2>&1 || pkg install -y libngtcp2 libnghttp3 >/dev/null 2>&1 || true
python -c "import flask" 2>/dev/null || pip install -q flask

if [ "$ROLE" != llm ]; then
  echo "==> PPE deps (numpy/pillow via pkg, onnxruntime via tur-repo)"
  pkg install -y tur-repo >/dev/null
  pkg install -y python-numpy python-pillow python-onnxruntime >/dev/null
fi

if [ "$ROLE" != ppe ]; then
  if python -c "import llama_cpp" 2>/dev/null; then
    echo "==> llama-cpp-python already installed"
  else
    echo "==> building llama-cpp-python (compiles on-device — expect ~10 min)"
    pkg install -y cmake clang ninja >/dev/null
    pip install llama-cpp-python
  fi
fi

echo "==> pulling worker code from central"
fetch "$CENTRAL/api/phone-brick/code.tar.gz" "$HOME/.phone-brick-code.tar.gz"
tar -xzf "$HOME/.phone-brick-code.tar.gz" -C "$HOME"
rm -f "$HOME/.phone-brick-code.tar.gz"

echo "==> writing ~/.phone-brick.env"
cat > "$HOME/.phone-brick.env" <<ENV
export PHONE_BRICK_CENTRAL="$CENTRAL"
export PHONE_BRICK_NAME="$NAME"
export WORKER_CENTRAL_URL="$CENTRAL"
export WORKER_NAME="$NAME-llm"
export PHONE_BRICK_ROLE="$ROLE"
ENV

mkdir -p "$HOME/bin" "$HOME/.termux/boot"
cat > "$HOME/bin/phone-brick-start" <<'START'
#!/data/data/com.termux/files/usr/bin/bash
# Start this phone's fleet workers. Idempotent: skips ones already running.
# Also installed as ~/.termux/boot/phone-brick.sh (runs at boot once the
# Termux:Boot app is installed and battery optimization is off for Termux).
. "$HOME/.phone-brick.env"
termux-wake-lock 2>/dev/null || true
pgrep -x sshd >/dev/null || sshd
cd "$HOME"
if [ "$PHONE_BRICK_ROLE" != llm ]; then
  pgrep -f "phone_brick worker" >/dev/null \
    || nohup python -m phone_brick worker >"$HOME/ppe-worker.log" 2>&1 &
fi
if [ "$PHONE_BRICK_ROLE" != ppe ]; then
  pgrep -f "\-m gguf_worker" >/dev/null \
    || nohup python -m gguf_worker >"$HOME/llm-worker.log" 2>&1 &
fi
START
chmod +x "$HOME/bin/phone-brick-start"
cp "$HOME/bin/phone-brick-start" "$HOME/.termux/boot/phone-brick.sh"

echo "==> starting workers"
"$HOME/bin/phone-brick-start"
sleep 5

echo "==> registration check"
if [ "$ROLE" != llm ]; then
  fetch "$CENTRAL/api/phone-brick/phones" | grep -q "\"$NAME\"" \
    && echo "    PPE worker '$NAME' registered with central" \
    || echo "    WARNING: '$NAME' not registered yet — check ~/ppe-worker.log"
fi
if [ "$ROLE" != ppe ]; then
  fetch "$CENTRAL/api/llm/workers" | grep -q "\"$NAME-llm\"" \
    && echo "    LLM worker '$NAME-llm' registered with central" \
    || echo "    WARNING: '$NAME-llm' not registered yet — check ~/llm-worker.log"
fi

cat <<'DONE'

Bootstrap complete. Remaining by-hand steps (these need the phone's screen/DeX):

  * Install the Termux:Boot app and open it once, then disable battery
    optimization for Termux — until both are done, the boot script
    (~/.termux/boot/phone-brick.sh) is inert and workers die with Termux.
  * PPE role: copy an ONNX model to ~/phone-brick/ppe-tanishjain-6class.onnx
    (or set MODEL_PATH in ~/.phone-brick.env). The worker runs without one
    but answers every run with nodet.
  * LLM role: assign a model from the console's workers panel — the worker
    self-downloads it from central. Keep it <=3B quantized on 8GB phones.
  * Optional ssh access for the ops box: add its key to
    ~/.ssh/authorized_keys (sshd listens on 8022; run 'passwd' first if you
    want password fallback).

Logs: ~/ppe-worker.log and ~/llm-worker.log. Re-run this script anytime to
update worker code or repair the setup.
DONE
