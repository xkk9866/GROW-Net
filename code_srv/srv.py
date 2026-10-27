"""Tiny SSH helper (retries on flaky tunnel):
python srv.py "<remote command>"   (runs in ~/research/grow_www27)
python srv.py get REMOTE LOCAL | put LOCAL REMOTE"""
import os  # public release: SSH configuration comes from the environment
import sys, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
import paramiko

HOST, PORT, USER, PWD = os.environ["SSH_HOST"], int(os.environ.get("SSH_PORT", "22")), os.environ["SSH_USER"], os.environ.get("SSH_PASSWORD")


def connect(tries=6):
    for k in range(tries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(4 + 4 * k)


def run(cmd, timeout=600):
    c = connect()
    _, o, e = c.exec_command("cd ~/research/grow_www27 && " + cmd, timeout=timeout)
    out, err = o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")
    c.close()
    return out + (("\n[stderr]\n" + err) if err.strip() else "")


def get(remote, local):
    c = connect(); s = c.open_sftp(); s.get(remote, local); s.close(); c.close()


def put(local, remote):
    c = connect(); s = c.open_sftp(); s.put(local, remote); s.close(); c.close()


if __name__ == "__main__":
    if sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "put":
        put(sys.argv[2], sys.argv[3])
    else:
        sys.stdout.write(run(sys.argv[1]))
