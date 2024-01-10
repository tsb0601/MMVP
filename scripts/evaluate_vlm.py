import csv
import os
from PIL import Image
import torch
from clip import load
import clip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv

def benchmark_model(model_name, benchmark_dir, device = "cpu"):
    model, preprocess = load(model_name, device=device)
    image_dir = os.path.join(benchmark_dir, 'MLLM_VLM Images')
    csv_file = os.path.join(benchmark_dir, 'Questions.csv')
    

    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    categories = [
        'Orientation and Direction', 'Presence of Specific Features', 
        'State and Condition', 'Quantity and Count', 
        'Positional and Relational Context', 'Color and Appearance',
        'Structural Characteristics', 'Texts',
        'Viewpoint and Perspective'
    ]

    pair_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
            img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            text1 = clip.tokenize([text1]).to(device)
            text2 = clip.tokenize([text2]).to(device)
            
            img1 = preprocess(img1).unsqueeze(0).to(device)
            img2 = preprocess(img2).unsqueeze(0).to(device)
            imgs = torch.cat((img1, img2), dim=0)


            with torch.no_grad():
                logits_per_image1, logits_per_text1 = model(imgs, text1)
                logits_per_image2, logits_per_text2 = model(imgs, text2)
                
                probs1 = logits_per_text1.softmax(dim=-1).cpu().numpy()
                probs2 = logits_per_text2.softmax(dim=-1).cpu().numpy()

            img1_score1 = probs1[0][0]
            img1_score2 = probs2[0][0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"

            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"

            
            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            current_category = categories[num_pairs // 15]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100

    return pair_accuracies


parser = argparse.ArgumentParser(description='Process a directory path.')
    
# Adding an argument for the directory path
parser.add_argument('--directory', type=str, help='The path to the directory')

# Parsing the arguments
args = parser.parse_args()

# OpenAI models
models = ['ViT-L/14']

results_openai = {f'openai-{model}': benchmark_model(model, args.directory) for model in models}


# Merge results
results = {**results_openai}

# Convert results to format suitable for star plot
categories = results[list(results.keys())[0]].keys()
data = {'Categories': list(categories)}
for model in list(results_openai.keys()):
    data[model] = [results[model][category] for category in categories]

print(results)

