from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.openicl.icl_evaluator import CircularEvaluator, AccEvaluator
from opencompass.datasets import MathBenchDataset, mathbench_postprocess
from opencompass.utils.text_postprocessors import first_option_postprocess

PROMPT_EN = {
    "FEWSHOT_INSTRUCTION_CLOZE" : """\
Examples:

<HUMAN>A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
<ASSISTANT>{thought} We need to calculate the total number of fruits. The total number of fruits in the first three baskets is given, while for the fourth basket, we need to subtract 2 from each fruit category. We can solve this problem using simple arithmetic.
{action} PythonInterpreter
{action_input}
```python
def solution():
    # Fruits in the first three baskets
    apples_first_three = 9
    oranges_first_three = 15
    bananas_first_three = 14

    # Fruits in the fourth basket
    apples_fourth = apples_first_three - 2
    oranges_fourth = oranges_first_three - 2
    bananas_fourth = bananas_first_three - 2

    # Total fruits
    total_fruits = ((apples_first_three + oranges_first_three + bananas_first_three) * 3 +
                    apples_fourth + oranges_fourth + bananas_fourth)

    return {{"total_fruits": total_fruits}}
```
<SYSTEM>{response}{{'total_fruits': 146}}
<ASSISTANT> {thought} By adding the given numbers of apples, oranges, and bananas in the first three baskets, then subtracting 2 from each category for the fourth basket, we have found the total number of fruits is 146.
{finish} 146

<HUMAN>Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
<ASSISTANT>{thought} This is a problem that requires solving equations. We know the relationship between the number of marbles, frisbees, and deck cards. Bella has twice as many marbles as frisbees, and 20 more frisbees than deck cards. Finally, we are told Bella buys 2/5 times more of each item. This purchasing will increase the number of each type of item.
{action} PythonInterpreter
{action_input}
```python
def solution():
    # Given number of marbles
    marbles_now = 60

    # Calculate number of frisbees and deck cards now
    frisbees_now = marbles_now / 2
    cards_now = frisbees_now - 20

    # Calculate number of each item after buying more
    marbles_then = marbles_now + (2/5) * marbles_now
    frisbees_then = frisbees_now + (2/5) * frisbees_now
    cards_then = cards_now + (2/5)*cards_now

    # Total number of items then
    total_items = marbles_then + frisbees_then + cards_then

    return {{"total_items": total_items}}
```
<SYSTEM>{response}{{'total_items': 140.0}}
<ASSISTANT>{thought} By establishing the relationships between the numbers of marbles, frisbees, and deck cards that Bella currently has, we can calculate how many of each item she will have after buying 2/5 more of each. Adding these quantities together gives us the total number of items.
{finish} 140
""",
    "FEWSHOT_INSTRUCTION_CHOICE" : """
Examples:

<HUMAN>Given point P(-1,4) lies on the graph of the inverse proportionality function $y=\\frac{{k}}{{x}}$ (k≠0), what is the value of k?
A. $-\\frac{{1}}{{4}}$
B. $\\frac{{1}}{{4}}$
C. $4$
D. $-4$
<ASSISTANT>{thought} Thought: We can substitute point P into the function to solve for the value of k.
{action} PythonInterpreter
{action_input}
```python
from sympy import solve, symbols, Ne, Eq
def solution():
    k = symbols('k')
    result = solve([Eq(4, k / (-1)), Ne(k, 0)], k, dict=True)
    return result
```
<SYSTEM>{response}{{Eq(k, -4)}}
<ASSISTANT> {thought} Substituting point P into the function yields the value of k as $-4$.
{finish} D

<HUMAN>The graph of the power function $y=(x)$ passes through the point$ (2, \\dfrac {{1}}{{4}}) $, what is the value of $f(-3)$?
A. $\\frac{{1}}{{9}}$
B. $\\frac{{1}}{{8}})=196-x$
C. $\\frac{{2}}{{9}}$
D. $\\frac{{1}}{{4}}$
<ASSISTANT>{thought} The question asks for the value of the power function at $x=-3$. To do this, we need to first determine the functional expression of the power function and then solve for the value.
{action} PythonInterpreter
{action_input}
```python
from sympy import *
def solution():
    # Define symbolic variables x, y, and k
    x, y, k = symbols('x y k')
    # Define the first equation: 2^k = 1/4
    eq1 = Eq(2**k, Rational(1, 4))
    # Solve for k in the equation
    k_value = solve(eq1, k)[0]
    # Define a symbolic expression for y in terms of x and k
    y = x**k_value
    # Substitute x with -3 in the expression for y
    result = y.subs(x, -3)
    # Return the result, which is the value of y when x is -3
    return result
```
<SYSTEM>{response}{{1/9}}
<ASSISTANT>{thought} The functional expression of the power function is $y=x^{{-2}}$. Substituting $x=-3$ yields $y=$\\frac{{1}}{{9}}$.
{finish} A
"""
}

PROMPT_CN = {
"FEWSHOT_INSTRUCTION_CLOZE" : """\
示例：

<HUMAN>一个有4个水果篮子，前三个篮子里有9个苹果、15个橙子和14个香蕉，第四个篮子里每种水果都少2个。总共有多少水果？
<ASSISTANT>{thought} 我们需要计算水果的总数。前三个篮子中的水果总数已知，而对于第四个篮子，我们需要从每种水果类别中减去2个。我们可以使用简单的算术来解决这个问题。
{action} PythonInterpreter
{action_input}
```python
def solution():
    # 前三个篮子中的水果
    apples_first_three = 9
    oranges_first_three = 15
    bananas_first_three = 14

    # 第四个篮子中的水果
    apples_fourth = apples_first_three - 2
    oranges_fourth = oranges_first_three - 2
    bananas_fourth = bananas_first_three - 2

    # 总水果数量
    total_fruits = ((apples_first_three + oranges_first_three + bananas_first_three) * 3 +
                    apples_fourth + oranges_fourth + bananas_fourth)

    return {{"total_fruits": total_fruits}}
```
<SYSTEM>{response}{{'total_fruits': 146}}
<ASSISTANT> {thought} 通过将前三个篮子中的苹果、橙子和香蕉的数量相加，然后从每个类别中减去2，我们找到了水果的总数为：146。
{finish} 146

<HUMAN>Bella有两倍于飞盘的弹珠。她还比卡片多20个飞盘。如果她买每种物品多2/5，她会有多少总数的物品，如果她现在有60颗弹珠？
<ASSISTANT>{thought} 这是一个需要解方程的问题。我们知道弹珠、飞盘和卡片之间的关系。Bella的弹珠数量是飞盘数量的两倍，比卡片多20个。最后，我们被告知Bella购买的每种物品都增加了2/5。这种购买会增加每种物品的数量。
{action} PythonInterpreter
{action_input}
```python
def solution():
    # 已知数量的弹珠
    marbles_now = 60

    # 计算现在飞盘和卡片的数量
    frisbees_now = marbles_now / 2
    cards_now = frisbees_now - 20

    # 计算购买更多后每种物品的数量
    marbles_then = marbles_now + (2/5) * marbles_now
    frisbees_then = frisbees_now + (2/5) * frisbees_now
    cards_then = cards_now + (2/5)*cards_now

    # 总物品数量
    total_items = marbles_then + frisbees_then + cards_then

    return {{"total_items": total_items}}
```
<SYSTEM>{response}{{'total_items': 140.0}}
<ASSISTANT>{thought} 通过建立Bella目前拥有的弹珠、飞盘和卡片数量之间的关系，我们可以计算出在购买每种物品多2/5后她将拥有多少个每种物品。将这些数量相加，得到总物品数量为140。
{finish} 140
""",

"FEWSHOT_INSTRUCTION_CHOICE" : """\
样例:

<HUMAN>已知点P（-1，4）在反比例函数$y=\\frac{{k}}{{x}}$ (k≠0)的图象上，则k的值是____
A. $-\\frac{{1}}{{4}}$
B. $\\frac{{1}}{{4}}$
C. $4$
D. $-4$
<ASSISTANT>{thought} Thought: 可以将点 P 带入函数解出 k 的值
{action} PythonInterpreter
{action_input}
```python
from sympy import solve, symbols, Ne, Eq
def solution():
    # 定义符号变量 k
    k = symbols('k')
    # 解决以下方程组:
    # 1. 4 = k / (-1)
    # 2. k != 0
    # 返回字典形式的结果
    result = solve([Eq(4, k / (-1)), Ne(k, 0)], k, dict=True)
    return result
```
<SYSTEM>{response}{{Eq(k, -4)}}
<ASSISTANT> {thought}将点 P 带入函数解出 k 的值为 $-4$
{finish} D

<HUMAN>幂函数$ y=(x) $的图象经过点$ (2, \\dfrac {{1}}{{4}}) $，则$ f(-3) $的值为 ______ ．
A. $\\frac{{1}}{{9}}$
B. $\\frac{{1}}{{8}})=196-x$
C. $\\frac{{2}}{{9}}$
D. $\\frac{{1}}{{4}}$
<ASSISTANT>{thought} 题目要求计算幂函数在 $x=-3$ 处的函数值，需要先求出幂函数的函数表达式再求解
{action} PythonInterpreter
{action_input}
```python
from sympy import *
def solution():
    # 定义符号变量 x、y 和 k
    x, y, k = symbols('x y k')
    # 定义第一个方程：2^k = 1/4
    eq1 = Eq(2**k, Rational(1, 4))
    # 解方程以找到 k 的值
    k_value = solve(eq1, k)[0]
    # 用 x 和 k 定义一个符号表达式 y
    y = x**k_value
    # 在表达式中将 x 替换为 -3
    result = y.subs(x, -3)
    # 返回结果，即当 x 为 -3 时的 y 的值
    return result
```
<SYSTEM>{response} 1/9
<ASSISTANT>{thought}求出幂函数的函数表达式为 $y=x^{{-2}}$，代入 $x=-3$ 得到 $y=$\\frac{{1}}{{9}}$
{finish} A
""",
}

mathbench_sets = {
    'college': ['single_choice_cn', 'cloze_en'],
    'high': ['single_choice_cn', 'single_choice_en'],
    'middle': ['single_choice_cn'],
    'primary': ['cloze_cn']
}

# Use circular evaluation or not
with_circular_eval = True

mathbench_datasets = []

for _split in list(mathbench_sets.keys()):
    for _name in mathbench_sets[_split]:
        prompt_example = PROMPT_CN if '_cn' in _name else PROMPT_EN
        mathbench_infer_cfg = dict(
            prompt_template=dict(type=PromptTemplate, template='{question}'),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=AgentInferencer, example=prompt_example["FEWSHOT_INSTRUCTION_CLOZE"] if 'cloze' in _name else prompt_example["FEWSHOT_INSTRUCTION_CHOICE"])
        )

        mathbench_eval_cfg = dict(
            evaluator=dict(type=CircularEvaluator if 'choice' in _name else AccEvaluator),
            pred_postprocessor=dict(type=first_option_postprocess, options='ABCD') if 'single_choice' in _name else dict(type=mathbench_postprocess, name=_name))

        mathbench_datasets.append(
            dict(
                type=MathBenchDataset,
                path=f"./data/mathbench/{_split}",
                name=_name,
                with_circular=with_circular_eval,
                abbr="mathbench-interpreter" + _split + '-' + _name,
                reader_cfg=dict(
                    input_columns=["question"],
                    output_column="answer"
                    ),
                infer_cfg=mathbench_infer_cfg,
                eval_cfg=mathbench_eval_cfg,
            ))
