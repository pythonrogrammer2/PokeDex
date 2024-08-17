import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import PIL.Image as Image

classes = ["Abra",
"Aerodactyl",
"Alakazam",
"Alolan Sandslash",
"Arbok",
"Arcanine",
"Articuno",
"Beedrill",
"Bellsprout",
"Blastoise",
"Bulbasaur",
"Butterfree",
"Caterpie",
"Chansey",
"Charizard",
"Charmander",
"Charmeleon",
"Clefable",
"Clefairy",
"Cloyster",
"Cubone",
"Dewgong",
"Diglett",
"Ditto",
"Dodrio",
"Doduo",
"Dragonair",
"Dragonite",
"Dratini",
"Drowzee",
"Dugtrio",
"Eevee",
"Ekans",
"Electabuzz",
"Electrode",
"Exeggcute",
"Exeggutor",
"Farfetchd",
"Fearow",
"Flareon",
"Gastly",
"Gengar",
"Geodude",
"Gloom",
"Golbat",
"Goldeen",
"Golduck",
"Golem",
"Graveler",
"Grimer",
"Growlithe",
"Gyarados",
"Haunter",
"Hitmonchan",
"Hitmonlee",
"Horsea",
"Hypno",
"Ivysaur",
"Jigglypuff",
"Jolteon",
"Jynx",
"Kabuto",
"Kabutops",
"Kadabra",
"Kakuna",
"Kangaskhan",
"Kingler",
"Koffing",
"Krabby",
"Lapras",
"Lickitung",
"Machamp",
"Machoke",
"Machop",
"Magikarp",
"Magmar",
"Magnemite",
"Magneton",
"Mankey",
"Marowak",
"Meowth",
"Metapod",
"Mew",
"Mewtwo",
"Moltres",
"MrMime",
"Muk",
"Nidoking",
"Nidoqueen",
"Nidorina",
"Nidorino",
"Ninetales",
"Oddish",
"Omanyte",
"Omastar",
"Onix",
"Paras",
"Parasect",
"Persian",
"Pidgeot",
"Pidgeotto",
"Pidgey",
"Pikachu",
"Pinsir",
"Poliwag",
"Poliwhirl",
"Poliwrath",
"Ponyta",
"Porygon",
"Primeape",
"Psyduck",
"Raichu",
"Rapidash",
"Raticate",
"Rattata",
"Rhydon",
"Rhyhorn",
"Sandshrew",
"Sandslash",
"Scyther",
"Seadra",
"Seaking",
"Seel",
"Shellder",
"Slowbro",
"Slowpoke",
"Snorlax",
"Spearow",
"Squirtle",
"Starmie",
"Staryu",
"Tangela",
"Tauros",
"Tentacool",
"Tentacruel",
"Vaporeon",
"Venomoth",
"Venonat",
"Venusaur",
"Victreebel",
"Vileplume",
"Voltorb",
"Vulpix",
"Wartortle",
"Weedle",
"Weepinbell",
"Weezing",
"Wigglytuff",
"Zapdos",
"Zubat"
]

full_model = torch.load("model_best_checkpoint.pth.tar")

model = full_model["model"]

resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 150
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet18_model.load_state_dict(model)

print(resnet18_model)

mean = [0.6053, 0.5874, 0.5538]
std = [0.2468, 0.2372, 0.2453]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)

    _, predicted = torch.max(output.data, 1)

    #print(predicted)

    print(classes[predicted.item()])
os.chdir("/Users/Owner/OneDrive/Desktop/images/eval/")
for picture in os.listdir():
    classify(resnet18_model, image_transforms, picture, classes)