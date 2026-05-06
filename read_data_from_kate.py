#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from get_filepaths import DATA_FOLDER




# =========================
# NAME HANDLING
# =========================

def base_sample_name(name):
    name = re.sub(r'(?<!\s)(mg_ml)', r' \1', name)
    name = re.sub(r'(C12E6|TX100|DDAC)(?!\s)', r'\1 ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    parts = name.split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]

    return " ".join(parts)


def resolve_sample_name(base_name, name_map, interactive=True):
    if base_name in name_map:
        return name_map[base_name]

    if not interactive:
        name_map[base_name] = base_name
        return base_name

    print(f"\nDetected sample: '{base_name}'")
    new_name = input("Rename? (Enter = keep, or type new name): ").strip()

    final_name = new_name if new_name else base_name
    name_map[base_name] = final_name
    return final_name


def load_name_map(mapping_file):
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_name_map(mapping_file, name_map):
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2)


# =========================
# CORE FUNCTION
# =========================

def rewrite_sample_names(input_path, output_path, name_map, interactive=True):
    output_lines = []

    with open(input_path, "r", encoding="latin1") as f:
        header = f.readline()
        output_lines.append(header)

        columns = header.strip().split("\t")
        if "Sample Name" not in columns:
            raise RuntimeError("Could not find 'Sample Name' column")

        sample_idx = columns.index("Sample Name")

        for line in f:
            parts = line.rstrip("\n").split("\t")

            # Skip malformed rows
            if len(parts) <= sample_idx:
                output_lines.append(line)
                continue

            original_name = parts[sample_idx]

            if original_name.strip():
                base = base_sample_name(original_name)

                new_base = resolve_sample_name(
                    base,
                    name_map,
                    interactive=interactive
                )

                # Preserve suffix (e.g. replicate numbers)
                suffix = original_name[len(base):]
                new_name = new_base + suffix

                parts[sample_idx] = new_name

            output_lines.append("\t".join(parts) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)


# =========================
# ENTRY POINT
# =========================

def rename_file(input_filename, output_filename=None, interactive=True):
    input_path = DATA_FOLDER / input_filename

    if output_filename is None:
        output_filename = input_path.stem + "_renamed.txt"

    output_path = DATA_FOLDER / output_filename

    mapping_file = DATA_FOLDER / "sample_name_map.json"
    name_map = load_name_map(mapping_file)

    print(f"Processing: {input_path.name}")

    rewrite_sample_names(
        input_path=input_path,
        output_path=output_path,
        name_map=name_map,
        interactive=interactive
    )

    save_name_map(mapping_file, name_map)

    print(f"Saved: {output_path.name}")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    # Example usage:
    rename_file("data_from_kate.txt")