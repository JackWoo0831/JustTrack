import argparse

from lagent import GPTAPI, ActionExecutor, ReAct, ReWOO
from lagent.actions import PythonInterpreter
from prompt_toolkit import ANSI, prompt

from agentlego.apis import load_tool

from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.agents.internlm2_agent import Internlm2Agent
from lagent.agents import internlm2_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        default=['ReferringTracker', 'ObjectDetection'],
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = HFTransformer(path='/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b', meta_template=META)

    tools = [load_tool(tool_type, device='cuda').to_lagent() for tool_type in args.tools]
    chatbot = Internlm2Agent(
        llm=model,
        max_turn=2,
        plugin_executor=ActionExecutor(actions=tools),
        protocol=internlm2_agent.Internlm2Protocol(
            tool=dict(
                begin='{start_token}{name}\n',
                start_token='<|action_start|>',
                name_map=dict(plugin='<|plugin|>', interpreter='<|interpreter|>'),
                belong='assistant',
                end='<|action_end|>\n',
            ),
        ),
    )

    while True:
        try:
            user = prompt(ANSI('\033[92mUser\033[0m: '))
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if user == 'exit':
            exit(0)

        result = chatbot.chat(user)
        for history in result.inner_steps:
            if history['role'] == 'system':
                print(f"\033[92mSystem\033[0m:{history['content']}")
            elif history['role'] in ['assistant', 'language', 'environment']:
                print(f"\033[92mBot\033[0m:\n{history['content']}")
            elif history['role'] in ['tool']:
                print(f"\033[92mTool\033[0m:\n{history['content']}")


if __name__ == '__main__':

    # tool = load_tool('SegmentObject', device='cuda')
    # segmentation = tool('M0101/img000001.jpg', 'bus')

    main()
    # Please detect the bus in the image `example.jpg`.
    # Please track the bus in the image sequence `M0101`
    # Please segment the laptop and cup in the video `test.mp4`