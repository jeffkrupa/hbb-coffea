import os
import subprocess
import json


def get_children(parent):
    #print(f"DEBUG : Call to get_children({parent})")
    command = f"eos root://cmseos.fnal.gov ls -F {parent}"
    # print(command)
    result = subprocess.getoutput(command)  # , stdout=subprocess.PIPE)
#    print(result)
    return result.split("\n")


def get_subfolders(parent):
    subfolders = []
    for x in get_children(parent):
        if len(x) == 0:
            continue
        if x[-1] == "/":
            subfolders.append(x)
    return subfolders


folders_to_index = [
    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/ttHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/ttHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/ttHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/ttHToBB",

    "/store/user/lpcpfnano/dryu/v2_2_1/2016/SingleMu2016",
    "/store/user/lpcpfnano/dryu/v2_2_1/2017/SingleMu2017",
    "/store/user/lpcpfnano/dryu/v2_2/2018/SingleMu2018",

    "/store/user/lpcpfnano/jekrupa/v2_2/2016/JetHT2016",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/JetHT2017",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/JetHT2018",

    #	"/store/user/lpcpfnano/cmantill/v2_2/2016/MET2016",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2017/MET2017",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2018/MET2018",

    #	"/store/user/lpcpfnano/drankin/v2_2/2016/SingleElectron2016",
    #	"/store/user/lpcpfnano/drankin/v2_2/2017/SingleElectron2017",
    #	"/store/user/lpcpfnano/drankin/v2_2/2018/SingleElectron2018",

    #	"/store/user/lpcpfnano/drankin/v2_2/2016/EGamma2016",
    #	"/store/user/lpcpfnano/drankin/v2_2/2017/EGamma2017",
    #	"/store/user/lpcpfnano/drankin/v2_2/2018/EGamma2018",

    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/emoreno/v2_2/2018/TTbar",

    "/store/user/lpcpfnano/drankin/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2018/TTbar",

    "/store/user/lpcpfnano/yihan/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2018/TTbar",
    
    "/store/group/lpcpfnano/jdickins/v2_3/2016APV/TTbarBoosted",
    "/store/group/lpcpfnano/jdickins/v2_3/2016/TTbarBoosted",
    "/store/group/lpcpfnano/jdickins/v2_3/2017/TTbarBoosted",
    "/store/group/lpcpfnano/jdickins/v2_3/2018/TTbarBoosted",

    "/store/user/lpcpfnano/yihan/v2_2/2016/QCD",
    "/store/user/lpcpfnano/yihan/v2_2/2016APV/QCD",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/QCD",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/QCD",

    "/store/user/lpcpfnano/jekrupa/v2_2/2016/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/WJetsToQQ",

    "/store/user/lpcpfnano/jekrupa/v2_2/2016/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/ZJetsToQQ",

    "/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop",
    "/store/user/lpcpfnano/dryu/v2_2_1/2016APV/SingleTop",
    "/store/user/lpcpfnano/pharris/v2_2/2017/SingleTop",
    "/store/user/lpcpfnano/pharris/v2_2/2018/SingleTop",

    "/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu",

    "/store/user/lpcpfnano/cmantill/v2_2/2016/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2016APV/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2017/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2018/DYJetsToLL",

    #	"/store/user/lpcpfnano/cmantill/v2_2/2016/HWW",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2016APV/HWW",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2017/HWW",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2018/HWW",

    #	"/store/user/lpcpfnano/cmantill/v2_2/2016/HTT",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2016APV/HTT",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2017/HTT",
    #	"/store/user/lpcpfnano/cmantill/v2_2/2018/HTT",

    #	"/store/user/lpcpfnano/dryu/v2_2/2016/VectorZPrime",
    #	"/store/user/lpcpfnano/jkrupa/v2_2/2016APV/VectorZPrime",
    #	"/store/user/lpcpfnano/jkrupa/v2_2/2017/VectorZPrime",
    #	"/store/user/lpcpfnano/dryu/v2_2/2018/VectorZPrime",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/Diboson",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/Diboson",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/Diboson",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/Diboson",
    
    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/DibosonNLO",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/DibosonNLO",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/DibosonNLO",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/DibosonNLO",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/VBFHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/VBFHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/VBFHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/VBFHToBB",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/EWKV",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/EWKV",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/EWKV",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/EWKV",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/GluGluHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/GluGluHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/GluGluHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/GluGluHToBB",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/WHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/WHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/WHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/WHToBB",

    "/store/group/lpcpfnano/jdickins/v2_2/2016APV/ZHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2016/ZHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2017/ZHToBB",
    "/store/group/lpcpfnano/jdickins/v2_2/2018/ZHToBB",

]

skip_list = []
skip_list = [os.path.normpath(x) for x in skip_list]

# Data path:
# .......................f1........................|...f2.....|..........f3.......|.....f4......|.f5.|....
# /store/user/lpcpfnano/dryu/v2_2/2017/SingleMu2017/SingleMuon/SingleMuon_Run2017C/211102_162942/0000/*root
#
# MC path:
# .......................f1........................|.......................f2..............................|..........f3.........|.....f4......|.f5.|....
# /store/user/lpcpfnano/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-800toInf/211108_171840/0000/*root

# /store/user/lpcpfnano/cmantill/v2_2/2016/HWW/
index = {}
for f1 in folders_to_index:
    f1 = f1.rstrip("/")
    year = f1.split("/")[-2]
    sample_short = f1.split("/")[-1]
    if not year in ["2016", "2016APV", "2017", "2018"]:
        raise ValueError(f"I couldn't find the year in path {f1}")

    if not year in index:
        index[year] = {}
    if not sample_short in index[year]:
        index[year][sample_short] = {}

    f1_subfolders = get_subfolders(f"{f1}")
    for f2 in f1_subfolders:
        print(f"\t/{f2}")
        # This should be the actual dataset name
        subsample_long = f2.replace("/", "")
        f2_subfolders = get_subfolders(f"{f1}/{f2}")

        for f3 in f2_subfolders:
            print(f"\t\t/{f3}")
            subsample_short = f3.replace("/", "")
            if not subsample_short in index[year][sample_short]:
                index[year][sample_short][subsample_short] = []
            f3_subfolders = get_subfolders(f"{f1}/{f2}/{f3}")

            if len(f3_subfolders) >= 2:
                print(
                    f"WARNING : Found multiple timestamps for {f1}/{f2}/{f3}:")
                # Remove anything in the skip list
                filtered_list = []
                this_skipped_list = []
                for f3_subfolder in f3_subfolders:
                    full_path_norm = os.path.normpath(
                        f"{f1}/{f2}/{f3}/{f3_subfolder}")
                    if not full_path_norm in skip_list:
                        filtered_list.append(f3_subfolder)
                    else:
                        this_skipped_list.append(full_path_norm)
                f3_subfolders = filtered_list

                if len(f3_subfolders) >= 2:
                    print(f3_subfolders)
                    print(
                        f"WARNING: I'm keeping them all, but double check and add bad folders to skip_list if needed.")
                else:
                    print(
                        f"WARNING: One timestamp remaining after filtering, all good. The following folders were skipped:")
                    print(this_skipped_list)

            for f4 in f3_subfolders:  # Timestamp
                f4_subfolders = get_subfolders(f"{f1}/{f2}/{f3}/{f4}")

                for f5 in f4_subfolders:  # 0000, 0001, ...
                    f5_children = get_children((f"{f1}/{f2}/{f3}/{f4}/{f5}"))
                    root_files = [f"{f1}/{f2}/{f3}/{f4}/{f5}/{x}".replace(
                        "//", "/") for x in f5_children if x[-5:] == ".root"]
                    index[year][sample_short][subsample_short].extend(
                        root_files)

with open("pfnanoindex.json", "w") as f:
    json.dump(index, f, sort_keys=True, indent=2)
