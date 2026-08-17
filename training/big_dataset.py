"""Generate large synthetic text dataset for training tests."""

import random


WORD_BANK = [
    "grug", "tribe", "hunt", "mammoth", "wolf", "fire", "cave", "river",
    "fish", "water", "star", "moon", "sun", "stone", "tool", "spear",
    "bear", "deer", "grass", "tree", "berry", "cliff", "mountain", "snow",
    "rain", "wind", "cold", "warm", "dark", "light", "elder", "child",
    "story", "song", "dance", "feast", "path", "track", "footprint", "blood",
    "meat", "bread", "flint", "bone", "hide", "fur", "camp", "night",
    "day", "dawn", "dust", "earth", "sky", "spirit", "ancestor", "shaman",
    "heal", "fight", "run", "walk", "sleep", "dream", "wake", "fear",
    "brave", "strong", "fast", "slow", "big", "small", "old", "young",
    "rock", "mud", "sand", "ice", "firelight", "smoke", "ash", "ember",
    "hunter", "gatherer", "cook", "maker", "tracker", "guard", "leader",
    "pack", "herd", "flock", "nest", "den", "lake", "stream", "forest",
    "plain", "valley", "ridge", "thunder", "lightning", "flood", "drought",
    "winter", "summer", "spring", "autumn", "seed", "root", "leaf", "branch",
    "flower", "fruit", "honey", "milk", "egg", "skin", "tooth", "claw",
    "horn", "antler", "tail", "wing", "eye", "ear", "nose", "mouth",
    "hand", "foot", "arm", "leg", "head", "back", "chest", "heart",
    "bone", "fur", "scale", "feather", "shell", "egg", "nest", "web",
    "grug", "tribe", "hunt", "mammoth", "wolf", "fire", "cave", "river",
    "fish", "water", "star", "moon", "sun", "stone", "tool", "spear",
    "bear", "deer", "grass", "tree", "berry", "cliff", "mountain", "snow",
    "rain", "wind", "cold", "warm", "dark", "light", "elder", "child",
    "story", "song", "dance", "feast", "path", "track", "footprint", "blood",
    "meat", "bread", "flint", "bone", "hide", "fur", "camp", "night",
    "day", "dawn", "dust", "earth", "sky", "spirit", "ancestor", "shaman",
    "heal", "fight", "run", "walk", "sleep", "dream", "wake", "fear",
    "brave", "strong", "fast", "slow", "big", "small", "old", "young",
    "rock", "mud", "sand", "ice", "firelight", "smoke", "ash", "ember",
    "hunter", "gatherer", "cook", "maker", "tracker", "guard", "leader",
    "pack", "herd", "flock", "nest", "den", "lake", "stream", "forest",
    "plain", "valley", "ridge", "thunder", "lightning", "flood", "drought",
    "winter", "summer", "spring", "autumn", "seed", "root", "leaf", "branch",
    "flower", "fruit", "honey", "milk", "egg", "skin", "tooth", "claw",
    "horn", "antler", "tail", "wing", "eye", "ear", "nose", "mouth",
    "hand", "foot", "arm", "leg", "head", "back", "chest", "heart",
    "bone", "fur", "scale", "feather", "shell", "egg", "nest", "web",
]


TEMPLATES = [
    "the {a} {verb} the {b} in the {place}",
    "{a} and {b} walk to {place} for {reason}",
    "{a} make {tool} from {material} near {place}",
    "in {place} the {a} {verb} while {b} watch",
    "{a} tell {child} story of {thing} and {thing2}",
    "when {weather} come the {a} go to {place}",
    "{a} track {b} by {sign} across {place}",
    "the {a} of {place} share {food} with {b}",
    "after {a} {verb} the {b} sleep in {place}",
    "{a} light {thing} so {b} can {verb} at night",
    "big {a} scare {b} away from {place}",
    "{a} climb {place} to find {thing} for {reason}",
    "the {a} hear {sound} from {place} and run",
    "{a} give {thing} to {b} because {reason}",
    "in cold {weather} {a} stay warm by {thing}",
    "{a} dream of {thing} while sleep in {place}",
    "{b} follow {a} path through {place} to {reason}",
    "the {a} taste {food} and {verb} with joy",
    "{a} hide from {b} behind {thing} in {place}",
    "at {time} the {a} wake and go to {place}",
    "{a} break {tool} on hard {thing} and {verb}",
    "{a} teach {b} how to {verb} near {place}",
    "river bring {thing} to {place} for {a}",
    "sky spirit send {weather} when {a} {verb}",
    "the {a} remember old {thing} from elder time",
    "{a} build strong {place} with {material} and {tool}",
    "{b} laugh when {a} fall into {thing}",
    "hunter {a} wait for {b} at {place} all {time}",
    "{a} mix {thing} and {thing2} to make {food}",
    "child {a} find {thing} and run to {b}",
]


VERBS = ["hunt", "run", "eat", "sleep", "fight", "dance", "sing", "walk",
         "track", "cook", "build", "hide", "climb", "wait", "watch", "make"]
PLACES = ["cave", "forest", "river", "plain", "mountain", "valley", "camp",
          "lake", "ridge", "cliff", "nest", "den", "tree", "grass"]
REASONS = ["food", "water", "safety", "warmth", "story", "rest", "hunt",
           "feast", "learn", "play"]
MATERIALS = ["stone", "bone", "wood", "flint", "hide", "mud", "ice", "ash"]
FOODS = ["meat", "fish", "berry", "bread", "honey", "egg", "root", "fruit"]
WEATHERS = ["rain", "snow", "wind", "cold", "storm", "dark", "sun"]
TIMES = ["day", "night", "dawn", "dusk", "moon"]
SOUNDS = ["roar", "howl", "thunder", "scream", "drum", "song"]
SIGNS = ["footprint", "smell", "sound", "broken branch", "droppings"]


def _fill(template: str, words: list) -> str:
    d = {
        "a": random.choice(words),
        "b": random.choice(words),
        "verb": random.choice(VERBS),
        "place": random.choice(PLACES),
        "reason": random.choice(REASONS),
        "tool": random.choice(["spear", "knife", "hammer", "rope", "bow"]),
        "material": random.choice(MATERIALS),
        "thing": random.choice(words),
        "thing2": random.choice(words),
        "child": random.choice(["child", "young", "cub", "pup"]),
        "weather": random.choice(WEATHERS),
        "food": random.choice(FOODS),
        "time": random.choice(TIMES),
        "sound": random.choice(SOUNDS),
        "sign": random.choice(SIGNS),
    }
    return template.format(**d)


def generate_dataset(num_stories: int = 1024, min_len: int = 128,
                     max_len: int = 1024, seed: int = 42) -> list:
    """Generate a big list of synthetic stories."""
    random.seed(seed)
    stories = []
    for _ in range(num_stories):
        parts = []
        while len(" ".join(parts)) < min_len:
            template = random.choice(TEMPLATES)
            sentence = _fill(template, WORD_BANK)
            # capitalize first letter, add period
            sentence = sentence[0].upper() + sentence[1:] + "."
            parts.append(sentence)
        story = " ".join(parts)
        story = story[:max_len]
        stories.append(story)
    return stories


if __name__ == "__main__":
    data = generate_dataset(10, 128, 256)
    for i, story in enumerate(data):
        print(f"--- story {i} ---")
        print(story)
        print(f"length: {len(story)}\n")
