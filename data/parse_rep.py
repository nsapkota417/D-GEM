import re

mode = 'PE'

if mode =='NE':
    input_txt = "/groups/dchen/mxs/ne_256_selection_results.txt"
    output_txt = "/users/nsapkota/VOS/data/datasets/cnh/indices.txt"

    with open(input_txt, "r") as f:
        content = f.read()

    # extract all integers
    indices = re.findall(r'\d+', content)

    # OPTIONAL: remove header numbers by only keeping the long list
    # (keeps numbers after "Candidate indices")
    if "Candidate indices" in content:
        content = content.split("Candidate indices")[-1]
        indices = re.findall(r'\d+', content)

    # write one per line
    with open(output_txt, "w") as f:
        for idx in indices:
            f.write(idx + "\n")

    print(f"Saved {len(indices)} indices to {output_txt}")
    
else:

    def process_indices(input_file, output_file):
        indices = []

        with open(input_file, 'r') as f:
            for line in f:
                if "Candidate indices" in line:
                    # extract numbers inside brackets
                    nums = re.findall(r'\d+', line)
                    indices = [int(x) + 1 for x in nums]
                    break

        if not indices:
            print("No indices found.")
            return

        # write to output file
        with open(output_file, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")

        print(f"Saved {len(indices)} indices to {output_file}")


    if __name__ == "__main__":
        input_path = "/groups/dchen/mxs/PE_dgem_task/selection_1pct/selection_results.txt"
        output_path = "/users/nsapkota/VOS/data/datasets/cnh_pe/indices.txt"

        process_indices(input_path, output_path)