import json
import os
import numpy as np


threshold = 0.5

protocols = [
    "icmo",
    "oimc",
    "ocim",
    "ocmi",
]


def idiap(sota):
    dataset = "idiap"
    for version in ["v1", "v2"]:
        print(f"Version: {version}")
        for protocol in protocols:
            print(f"\tProtocol: {protocol}")
            for attack in ["display", "print", "real"]:
                fname = f"./results/{dataset}/{sota}/{protocol}/{attack}_{version}.json"
                with open(fname) as f:
                    data = json.load(f)

                results = data["result"]
                wrong = 0
                for result in results:
                    if attack == "real":
                        if result < threshold:
                            wrong += 1
                    else:
                        if result > threshold:
                            wrong += 1

                print(
                    f"\t\t{attack.capitalize()} Not passing: {wrong}/{len(results)}={wrong * 100 / len(results):.3f}%"
                )


def hda(sota):
    dataset = "hda"
    for protocol in protocols:
        fname = f"./results/{dataset}/{sota}/{protocol}"
        print(f"Protocol: {protocol}")
        for exp in os.listdir(fname):
            fname = f"./results/{dataset}/{sota}/{protocol}/{exp}"
            with open(fname) as f:
                data = json.load(f)

            results = data["result"]
            wrong = 0
            for result in results:
                if exp == "real":
                    if result <= threshold:
                        wrong += 1
                else:
                    if result > threshold:
                        wrong += 1

            print(
                f"\t{exp} Not passing: {wrong}/{len(results)}={wrong * 100 / len(results):.3f}%"
            )


def nexdata(sota):
    dataset = "nexdata"
    for protocol in protocols:
        print(f"Protocol: {protocol}")
        for exp in [
            "display_ipad",
            "display_mobilephone",
            "print_paper_mask",
            "print_paper_photo",
            "real",
        ]:
            fname = f"./results/{dataset}/{sota}/{protocol}/{exp}.json"
            with open(fname) as f:
                data = json.load(f)

            results = data["result"]
            wrong = 0
            for result in results:
                if exp == "real":
                    if result <= threshold:
                        wrong += 1
                else:
                    if result > threshold:
                        wrong += 1

            print(
                f"\t{exp} Not passing: {wrong}/{len(results)}={wrong * 100 / len(results):.3f}%"
            )


def trainingProIBeta2(sota):
    dataset = "trainingproibeta2"
    for protocol in protocols:
        print(f"Protocol: {protocol}")
        for exp in ["latex", "silicon", "wrap", "real"]:
            fname = f"./results/{dataset}/{sota}/{protocol}/{exp}.json"
            with open(fname) as f:
                data = json.load(f)

            results = data["result"]
            wrong = 0
            for result in results:
                if exp == "real":
                    if result <= threshold:
                        wrong += 1
                else:
                    if result > threshold:
                        wrong += 1

            print(
                f"\t{exp} Not passing: {wrong}/{len(results)}={wrong * 100 / len(results):.3f}%"
            )


if __name__ == "__main__":
    sota = "flip_fas"
    print("Idiap")
    idiap(sota)
    print(
        "------------------------------------------------------------------------------------------"
    )
    print("HDA")
    hda(sota)
    print(
        "------------------------------------------------------------------------------------------"
    )
    print("Nexdata")
    nexdata(sota)
    print(
        "------------------------------------------------------------------------------------------"
    )
    print("TrainingProIBeta2")
    trainingProIBeta2(sota)
    print(
        "------------------------------------------------------------------------------------------"
    )
