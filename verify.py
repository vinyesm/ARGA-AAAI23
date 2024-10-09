"""
A script to visually verify solutions
"""

import json
from typing import List

from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt

from image import Image


def verify(grid, abstraction, apply_call):
    i = Image("", grid)
    g = getattr(i, Image.abstraction_ops[abstraction])()
    for j, call in enumerate(apply_call):
        g.apply(**call)
    go = g.undo_abstraction()
    # print(f"M2: {[(n, go.graph.nodes[n]['color']) for n in go.graph.nodes]}")
    nodes = go.graph.nodes
    return to_grid(nodes, go.width, go.height)

def to_grid(nodes, w, h):
    g = [[0 for _ in range(w)] for _ in range(h)]
    for i, j in nodes:
        g[i][j] = nodes[(i,j)]['color']
    return g


def plot_task(
    ins: List[List[List[int]]],
    outs: List[List[List[int]]],
    tries: List[List[List[int]]],
    title: str = None,
    save_to_png: bool = False,
) -> None:
    """
    displays a task withs tried solutions tries
    """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    height = 3
    width = len(ins)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    if len(ins) == 1:
        axes[0].imshow(ins[0], **args)
        if outs: 
            axes[1].imshow(outs[0], **args)
        if tries: 
            axes[2].imshow(tries[0], **args)
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[0].text(0, 0.5, 'Input', fontsize=12, va='center', ha='right', transform=axes[0, column].transAxes)
        axes[1].text(0, 0.5, 'Output', fontsize=12, va='center', ha='right', transform=axes[1, column].transAxes)
        axes[2].text(0, 0.5, 'Tried Solution', fontsize=12, va='center', ha='right', transform=axes[2, column].transAxes)
    else:
        for column in range(len(ins)):
            axes[0, column].imshow(ins[column], **args)
            if column < len(outs): 
                axes[1, column].imshow(outs[column], **args)
            if column < len(tries): 
                axes[2, column].imshow(tries[column], **args)
            axes[0, column].axis('off')
            axes[1, column].axis('off')
            axes[2, column].axis('off')
            if column == 0:
                axes[0, column].text(0, 0.5, 'Input', fontsize=12, va='center', ha='right', transform=axes[0, column].transAxes)
                axes[1, column].text(0, 0.5, 'Output', fontsize=12, va='center', ha='right', transform=axes[1, column].transAxes)
                axes[2, column].text(0, 0.5, 'Tried Solution', fontsize=12, va='center', ha='right', transform=axes[2, column].transAxes)

            # Turn off axis ticks for all plots
            axes[0, column].axis('off')
            axes[1, column].axis('off')
            axes[2, column].axis('off')
    
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if title is not None:
        plt.savefig(f'image_{title}.png')
    else:
        plt.savefig('image.png')


if __name__ == "__main__":
    key = 'ddf7fa4f'
    with open(f'dataset/training/{key}.json') as f:
        task = json.load(f)

    with open(f'solutions/correct/solutions_{key}.json') as f:
        solution = json.load(f)

    # input grids
    ins = [x['input'] for x in task['train']] + [x['input'] for x in task['test']]
    # output grids (ground truth)
    outs = [x['output'] for x in task['train']]

    # computing guessed output grids given a found solution
    abstraction = solution['abstraction']
    apply_call = solution['apply_call']
    tries = [verify(gi, abstraction, apply_call) for gi in ins]

    # check if we have correct outputs
    print(f"checking if is solution correct: {outs == tries[0:len(outs)]}")

    plot_task(ins, outs, tries, f"{key}")

## to test on a single grid do:
# a_in = task['train'][0]['input']
# a_out = task['train'][0]['output']
# abstraction = solution['abstraction']
# apply_call = solution['apply_call']
# a_try = verify(a_in, abstraction, apply_call)
# plot_task([a_in], [a_out], [a_try])


