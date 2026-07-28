import os
import time
import socket
import urllib.request

TOPIC = "vmunoz-RELELA-06"

JOBS = [
    {
        "pid": 1785750,
        "name": "O02_mha_standard_hard100k_hidden384",
    },
    {
        "pid": 1787528,
        "name": "E20_mha_standard_hard100k_rsd03",
    },
    # {
    #     "pid": 1596359,
    #     "name": "O01_mha_standard_hard100k_hidden128",
    # }
]

CHECK_EVERY_SECONDS = 300  # 5 minutos


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def notify(message: str) -> None:
    url = f"https://ntfy.sh/{TOPIC}"
    data = message.encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Title": "Entrenamiento terminado",
            "Priority": "high",
            "Tags": "tada,gpu",
        },
    )

    with urllib.request.urlopen(request) as response:
        response.read()


def main():
    hostname = socket.gethostname()
    pending_jobs = JOBS.copy()

    print(f"Monitoreando {len(pending_jobs)} procesos...")
    for job in pending_jobs:
        print(f"PID {job['pid']}: {job['name']}")

    while pending_jobs:
        still_pending = []

        for job in pending_jobs:
            pid = job["pid"]
            name = job["name"]

            if process_is_running(pid):
                still_pending.append(job)
            else:
                message = f"{name} terminó en {hostname}. PID {pid} finalizó."
                print(message)
                notify(message)

        pending_jobs = still_pending

        if pending_jobs:
            time.sleep(CHECK_EVERY_SECONDS)

    print("Todos los entrenamientos terminaron.")


if __name__ == "__main__":
    main()