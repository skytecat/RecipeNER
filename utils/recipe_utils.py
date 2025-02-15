import matplotlib.pyplot as plt

import seaborn as sns

from ipymarkup         import show_box_markup
from ipymarkup.palette import PALETTE, BLUE, RED, GREEN, PURPLE, BROWN, ORANGE, material

from sklearn.metrics import confusion_matrix

import torch
from collections import Counter

PALETTE.set('NAME', BLUE)
PALETTE.set('UNIT', RED)
PALETTE.set('QTY', GREEN)
PALETTE.set('RANGE_END', GREEN)
PALETTE.set('INDEX', PURPLE)
PALETTE.set('COMMENT', ORANGE)
PALETTE.set('OTHER', BROWN)


def show_markup(recipe,tags):
    mapper = lambda tag: tag[2:] if tag!='OTHER' else tag
    
    tags  = [mapper(tag) for tag in tags]
    text  = ' '.join(recipe)
    spans = []
        
    start,end,tag = 0,len(recipe[0]),tags[0]
    
    for word, ttag in zip(recipe[1:], tags[1:]): 
        
        if tag == ttag:
            end  += 1 + len(word)
            
        else:
            span  = (start, end, tag)
            spans.append(span)
        
            start = 1 + end
            end  += 1 + len(word)
            tag   = ttag
            
    span  = (start, end, tag)
    spans.append(span) 
    
    # для отрисовки на темном фоне
    colors = BLUE, RED, GREEN, PURPLE, BROWN, ORANGE

    for c in colors:
        color_name = c.name[0].capitalize() + c.name[1:]
        c.background = material(color_name, '900')         
            
    show_box_markup(text, spans, palette=PALETTE)
    
    
def predict_tags(model, converter, recipe):
    
    encoded_recipe = converter.words_to_index(recipe)        # слово -> его номер в словаре
    
    encoded_tags   = model.predict_tags(encoded_recipe)      # предсказанные тэги (номера)

    decoded_tags   = converter.indices_to_tags(encoded_tags) # номер тэга -> тэг
    
    return decoded_tags
    
    
def tag_statistics(model, converter, data):
    def tag_counter(predicted, ground):
        correct_tags = Counter()
        ground_tags  = Counter(ground)
        
        for tag_p, tag_g in zip(predicted, ground):
            if tag_p == tag_g:
                correct_tags[tag_g] += 1
  
        return correct_tags, ground_tags
    
    
    total_correct, total_tags = Counter(), Counter()
    
    for recipe, tags in data:
        tags_pred              = predict_tags(model, converter, recipe)
        tags_correct, tags_num = tag_counter(tags_pred, tags)
       
        total_correct.update(tags_correct)
        total_tags.update(tags_num)

    return total_correct, total_tags


def plot_confusion_matrix(y_true, y_pred, classes):
    # Создаем два подграфика (1 строка, 2 столбца)
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, 1-й график
    cm = confusion_matrix(y_true, y_pred, labels=classes)
        
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.subplot(1, 2, 2)  # 1 строка, 2 столбца, 2-й график
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    
    sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, fmt='.2f')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Отображаем оба графика
    plt.tight_layout() 
    plt.show()


def form_vocabulary_and_tagset(recipes):
    dictionary = set()
    labels     = set()
    for line in recipes:
        if len(line) > 0:
            word, label = line.split('\t')
            dictionary.add(word)
            labels.add(label)
            
    return dictionary,labels


def prepare_data(lines):
    recipes_w_tags = []

    recipe, tags = [],[]
    
    for line in lines:
        # запись текущего рецепта
        if len(line) > 0:
            word, label = line.split('\t')
            recipe.append(word)
            tags.append(label)
        # конец рецепта, добавляем его в recipes_w_tags
        else:
            if len(recipe) > 0:
                recipes_w_tags.append((recipe,tags))
            recipe, tags = [],[]

    return recipes_w_tags


class Converter():
    def __init__(self,vocabulary, tags):
        self.words = sorted(vocabulary)
        self.tags  = sorted(tags)

        self.word_with_idx = { word:idx for idx,word in enumerate(self.words)}
        self.tag_with_idx  = { tag:idx for idx,tag  in enumerate(self.tags)}
        
    def words_to_index(self, words):
        return torch.tensor([self.word_with_idx[w] for w in words], dtype=torch.long)
    
    def tags_to_index(self, words):
        return torch.tensor([self.tag_with_idx[w] for w in words], dtype=torch.long)
    
    def indices_to_words(self, indices):
        return [self.words[i] for i in indices]
    
    def indices_to_tags(self, indices):
        return [self.tags[i] for i in indices]