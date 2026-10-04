from pathlib import Path
from electrode_detection.dataset import export_ls_json

refresh_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA5ODI2NDA2MCwiaWF0IjoxNzkxMDY0MDYwLCJqdGkiOiI0MDc1MTAyODkwMzY0ODZjYThmMmQzZWFkN2Q5YWVjOSIsInVzZXJfaWQiOiIxIn0.F2UiEua_akUxmCoSwtTYhVsfbQllJT27RlPbVyObMHU"
json_path = Path("datasets/side1/label-studio.json")
project_id = 19

def main() -> None:

    export_ls_json(
        refresh_token=refresh_token,
        out_path=json_path,
        project_id=project_id,
    )

if __name__ == "__main__":

    main()