"""Gera uma imagem PNG do fluxo de decisão para consulta local."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts" / "langgraph_flow.png"


def box(axis, x, y, label, color="#E8F1FA"):
    axis.add_patch(
        FancyBboxPatch(
            (x - 1.2, y - .28), 2.4, .56, boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="#315C84", linewidth=1.2,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=8, wrap=True)


def arrow(axis, start, end, label=""):
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, color="#315C84")
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .14,
            label, ha="center", fontsize=7, color="#315C84",
        )


def render(target: Path = TARGET) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")
    box(axis, 5, 6.4, "Entrada e anonimização")
    box(axis, 5, 5.2, "Triagem de risco")
    box(axis, 1.7, 4.0, "Alerta local", "#FDE9E7")
    box(axis, 5, 4.0, "Consulta estruturada")
    box(axis, 8.3, 4.0, "Recusa", "#FDE9E7")
    box(axis, 5, 2.8, "Recuperação de protocolos")
    box(axis, 3.0, 1.5, "Retenção segura", "#FFF4CC")
    box(axis, 7.0, 1.5, "Geração com LLM")
    box(axis, 7.0, .35, "Verificação e fontes")
    arrow(axis, (5, 6.12), (5, 5.5))
    arrow(axis, (4.3, 5.0), (2.1, 4.3), "sinal crítico")
    arrow(axis, (5, 4.9), (5, 4.3), "sem risco")
    arrow(axis, (5.7, 5.0), (7.9, 4.3), "pedido impróprio")
    arrow(axis, (5, 3.7), (5, 3.1))
    arrow(axis, (4.6, 2.5), (3.3, 1.8), "sem contexto")
    arrow(axis, (5.4, 2.5), (6.7, 1.8), "com contexto")
    arrow(axis, (7, 1.2), (7, .65))
    arrow(axis, (6.5, .2), (3.5, 1.2), "saída não aceita")
    figure.tight_layout()
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


if __name__ == "__main__":
    print(render().relative_to(ROOT).as_posix())
