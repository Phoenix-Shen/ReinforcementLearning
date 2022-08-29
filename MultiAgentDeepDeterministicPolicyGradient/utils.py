import yaml


def load_cfg(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

        print("###################YOUR SETTINGS###################")
        for key in args.keys():
            print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")
        return args
