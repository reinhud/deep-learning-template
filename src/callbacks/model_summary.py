from typing import Any, List, Tuple

from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import get_human_readable_count
from rich import box
from rich.text import Text


class CustomRichModelSummary(RichModelSummary):
    """Customized ModelSummary callback.

    This class extends the RichModelSummary callback to provide custom formatting
    and additional information in the model summary output.

    Args:
        max_depth (int, optional): Maximum depth to summarize the model layers.
            Set to 0 to turn off summary.
    """

    def __init__(self, max_depth: int = -1):
        super().__init__(max_depth=max_depth)

    @staticmethod
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        **summarize_kwargs: Any,
    ) -> None:
        """Prints a summary of the model.

        Args:
            summary_data (List[Tuple[str, List[str]]]): Data to summarize,
                including layer names and parameters.
            total_parameters (int): Total number of parameters in the model.
            trainable_parameters (int): Number of trainable parameters in the model.
            model_size (float): Estimated size of the model in megabytes.
            summarize_kwargs: Additional keyword arguments for customization.

        Returns:
            None
        """
        from rich import get_console
        from rich.table import Table

        console = get_console()

        # Define table columns
        table = Table(
            title=Text("Model Summary", style="bold green_yellow"),
            box=box.ROUNDED,
            header_style="green_yellow",
        )
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True, style="purple4")
        table.add_column("Type")
        table.add_column("Params", justify="right", style="purple4")

        # Add input and output size columns if available
        column_names = list(zip(*summary_data))[0]
        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        # Add rows to the table
        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        # Print the summary table
        console.print(table)

        # Print additional summary information
        parameters = []
        for param in [
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size,
        ]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: [bold green_yellow]{parameters[0]}[/]")
        grid.add_row(f"[bold]Non-trainable params[/]: [bold green_yellow]{parameters[1]}[/]")
        grid.add_row(f"[bold]Total params[/]: [bold green_yellow]{parameters[2]}[/]")
        grid.add_row(
            f"[bold]Total estimated model params size (MB)[/]: [bold green_yellow]{parameters[3]}[/]"
        )

        console.print(grid)
