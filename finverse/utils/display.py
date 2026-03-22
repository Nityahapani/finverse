from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()


def print_header(title: str, subtitle: str = ""):
    console.print()
    console.print(Panel(
        f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]" if subtitle else f"[bold white]{title}[/bold white]",
        border_style="blue",
        padding=(0, 2),
    ))


def print_table(title: str, rows: list[tuple], headers: list[str], style_fn=None):
    table = Table(title=title, box=box.SIMPLE_HEAD, show_header=True, header_style="bold blue")
    for h in headers:
        table.add_column(h, justify="right" if h != headers[0] else "left")
    for row in rows:
        styled = [str(v) for v in row]
        table.add_row(*styled)
    console.print(table)


def fmt_currency(v, decimals=1):
    if v is None:
        return "—"
    if abs(v) >= 1_000:
        return f"${v:,.{decimals}f}B"
    return f"${v:.{decimals}f}B"


def fmt_pct(v, decimals=1):
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_multiple(v, decimals=1):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}x"


def fmt_price(v):
    if v is None:
        return "—"
    return f"${v:.2f}"
