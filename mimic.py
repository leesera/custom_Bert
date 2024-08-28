import math
import random
from typing import List

labels = [
    ['lie', 'bed', 1.],
    ['sit', 'bed', 2.],
    ['awaken', 'bed', 3.],
    ['hold', 'book', 2.],
    ['put', 'book', 2.],
    ['take', 'book', 2.],
    ['close', 'book', 3.],
    ['open', 'book', 3.],
    ['use', 'book', 4.],
    ['sit', 'chair', 2.],
    ['stand', 'chair', 3.],
    ['hold', 'laptop', 2.],
    ['put', 'laptop', 2.],
    ['take', 'laptop', 2.],
    ['close', 'laptop', 3.],
    ['open', 'laptop', 3.],
    ['use', 'laptop', 4.],
    ['put', 'paper', 2.],
    ['take', 'paper', 2.],
    ['hold', 'paper', 3.],
    ['use', 'paper', 4.],
    ['put', 'phone', 2.],
    ['take', 'phone', 2.],
    ['hold', 'phone', 3.],
    ['use', 'phone', 4.],
    ['lie', 'sofa', 1.],
    ['sit', 'sofa', 2.],
    ['sit', 'somewhere', 2.],
    ['awaken', 'somewhere', 3.],
    ['stand', 'somewhere', 3.],
]

transitions = [
    ['stand', 'sit'],
    ['sit', 'stand'],
    ['sit', 'lie'],
    ['lie', 'awaken'],
    ['awaken', 'stand'],
    ['take', 'hold'],
    ['hold', 'open'],
    ['hold', 'use'],
    ['open', 'use'],
    ['use', 'close'],
    ['close', 'put'],
    ['put', 'take'],
]

def check_transition(p: str, q: str):
    for u, v in transitions:
        if u == p and v == q:
            return True
    return False

confidences = [
    'low',
    'medium',
    'high',
]

level = [
    'zero',
    'one',
    'two',
    'three',
    'four',
]

K = 3
F = 10

class intent():
    label: int
    confidence: int
    def __init__(self, label: int, confidence: int):
        self.label = label
        self.confidence = confidence

    def energy(self) -> float:
        return labels[self.label][2]
    
    def __str__(self) -> str:
        return f"{labels[self.label][0]} {labels[self.label][1]} {confidences[self.confidence]}"

class frame():
    intents: List[intent]
    occupy: bool
    def __init__(self, label: int):
        rnd = []
        for l in range(len(labels)):
            if l == label:
                p = random.gauss(0.70, 0.15)
            elif label != -1 and (labels[l][0] == labels[label][0] or labels[l][1] == labels[label][1]):
                p = random.gauss(0.30, 0.15)
            else:
                p = random.gauss(0.10, 0.15)
            rnd.append((p, l))
        rnd = sorted(rnd, key=lambda v: -v[0])

        self.intents = []
        self.occupy = label != -1
        chk = {}
        cnt = 0
        for k in range(len(labels)):
            prob, label = rnd[k]
            if labels[label][1] in chk:
                continue
            if prob >= 0.6:
                confidence = 2
            elif prob >= 0.3:
                confidence = 1
            else:
                confidence = 0
            self.intents.append(intent(label, confidence))
            chk[labels[label][1]] = True
            cnt += 1
            if cnt == K:
                break

    def energy(self) -> float:
        if not self.occupy:
            return 0.
        e = 0
        rat = 1
        sum = 0
        for intent in self.intents:
            rat *= 0.3
            e += rat * intent.energy()
            sum += rat
        e /= sum
        return e
    
    def __str__(self) -> str:
        ret = []
        if self.occupy:
            ret.append('occupy')
        else:
            ret.append('empty')
        for intent in self.intents:
            ret.append(intent.__str__())
        return ' '.join(ret)
    
class sentence():
    frames: List[frame]
    def __init__(self):
        self.frames = []
        l = random.randint(0, len(labels)) - 1
        for _ in range(F):
            self.frames.append(frame(l))
            if random.randint(1, 8) == 1:
                l = random.randint(0, len(labels)) - 1
            elif random.randint(1, 3) == 1:
                if l == -1:
                    l = random.randint(1, len(labels)) - 1
                else:
                    if random.randint(0, 4) == 0:
                        l = -1
                    else:
                        nxts = [l]
                        for n in range(len(labels)):
                            if check_transition(labels[l][0], labels[n][0]) and labels[l][1] == labels[n][1]:
                                nxts.append(n)
                        l = random.choice(nxts)

    def energy(self) -> float:
        e = 0
        rat = 1
        sum = 0
        for frame in self.frames[::-1]:
            rat *= 0.3
            e += rat * frame.energy()
            sum += rat
        e /= sum
        e = math.ceil(e)
        if e == 0:
            e = 1
        return e
    
    def __str__(self) -> str:
        ret = []
        for frame in self.frames:
            ret.append(frame.__str__())
            ret.append("eof")
        ret[-1] = "sep"
        ret.append(level[self.energy()])
        return ' '.join(ret)

for _ in range(50000):
    print(sentence())
    print()
