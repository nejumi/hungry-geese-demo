#!/usr/bin/env python
# coding: utf-8

from PIL import Image, ImageDraw, ImageColor
from kaggle_environments import make

from utils import state_to_board, obs_to_agent_input, load_agent_from_file

def create_board_image(board, cell_size=20, goose_colors=None, food_color='purple'):
    rows, cols = len(board), len(board[0])
    img = Image.new('RGB', (cols * cell_size, rows * cell_size), color='white')
    draw = ImageDraw.Draw(img)

    if goose_colors is None:
        goose_colors = ['red', 'green', 'blue', 'yellow']

    food_positions = []

    for row in range(rows):
        for col in range(cols):
            cell = board[row][col]
            if cell == -1:
                draw.rectangle([(col * cell_size, row * cell_size), ((col + 1) * cell_size, (row + 1) * cell_size)], fill='black')
            elif cell >= 0:
                base_color = ImageColor.getrgb(goose_colors[cell % len(goose_colors)])
                alpha = int(255 * (1 - (cell - int(cell))))
                color = tuple(int(c * alpha / 255) for c in base_color)
                draw.rectangle([(col * cell_size, row * cell_size), ((col + 1) * cell_size, (row + 1) * cell_size)], fill=color)
            elif cell == -2:
                food_positions.append((row, col))

    for pos in food_positions:
        row, col = pos
        draw.ellipse([(col * cell_size, row * cell_size), ((col + 1) * cell_size, (row + 1) * cell_size)], fill=food_color)

    return img

def game_to_gif(frames, output_path, duration=500, loop=0):
    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=loop)

def play_and_create_gif(agent, output_path, cell_size=20, goose_colors=None, duration=500, loop=0):
    env = make("hungry_geese")
    env.reset(num_agents=4)
    frames = []

    while not env.done:
        actions = [agent(obs_to_agent_input(env.state[i], env.state[0]), env.state) for i in range(4)]
        env.step(actions)
        board = state_to_board(env.state, env.configuration)
        img = create_board_image(board, cell_size=cell_size, goose_colors=goose_colors)
        frames.append(img)

    game_to_gif(frames, output_path, duration=duration, loop=loop)

def create_gif_from_submission(submission_path, output_gif_path):
    agent = load_agent_from_file(submission_path)
    play_and_create_gif(agent, output_gif_path)
