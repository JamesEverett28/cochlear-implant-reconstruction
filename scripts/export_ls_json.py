from pathlib import Path
from electrode_detection.dataset import export_ls_json

refresh_token = None
project_id = 19

json_path = Path("datasets/side1/label-studio.json")

def main() -> None:

    export_ls_json(
        refresh_token=refresh_token,
        out_path=json_path,
        project_id=project_id,
    )

if __name__ == "__main__":

    main()