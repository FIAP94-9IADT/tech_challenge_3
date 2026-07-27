"""Interface de linha de comando para demonstração do fluxo."""

from __future__ import annotations

import argparse

from .factory import build_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consulta clínica institucional assistida")
    parser.add_argument("--patient-id", required=True, help="Identificador sintético no formato PAC-0000")
    parser.add_argument("--question", required=True, help="Pergunta clínica sem identificadores pessoais")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = build_application().invoke(args.question, args.patient_id)
    print(state["answer"])
    print("\nEtapas executadas:")
    for step in state["steps"]:
        print(f"- {step}")


if __name__ == "__main__":
    main()
