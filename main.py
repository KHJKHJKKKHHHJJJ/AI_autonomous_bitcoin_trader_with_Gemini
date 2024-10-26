print("========= Welcome to GEMINI BTC TRADER =========")
print("initializing...")
import bit_AI
import time
from inputimeout import inputimeout
import json
import asyncio

def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)


print("Gemini will be processed.")
time.sleep(0.5)
print("If you want to stop the program, press Y")

bit_instruction = './Bitcoin Gemini Instruction.md'
bit_chat_session = bit_AI.gen_bit_model(get_instructions(bit_instruction))

former_dec = 3 # 1 = sell/ 0 = buy/ 3 = initial

u_input = ''
timeout= 60 * 60 * 1.5
while u_input not in ['Y', 'y', 'Yes', 'yes', 'YES']:
    print("Gemini is preprocessing...")
    try:
        decision = bit_AI.gem_sug(bit_chat_session, bit_AI.get_today_prudence())
        # for debugging
        # decision = dict()
        # decision['decision'] = 'sell'
        # decision['amount'] = '1000'
        # decision['ET'] = '1h 30m'
        # decision['profit'] = '0'
        # decision['reason'] = 'adsfadsfadfs'
    except Exception as e:
        print("Error occured :", e)
        print("restarting...")
        timeout = 10 * 60 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(bit_AI.tel(f"ERROR OCCURED:\t{e}"))
        if '429' in str(e):
            continue
        else:
            break
    try:
        if decision['decision'] == 'buy':
            print("Buying process will be proceeded")
            buy_sign = bit_AI.market_buy(amount = float(decision['amount']))

            if json.loads(buy_sign)['result'] != 'success':
                print("Error Occured: ", end='')
                print(json.loads(buy_sign)['error_code'])

        elif decision['decision'] == 'sell':
            print("Selling process will be proceeded")
            sell_sign = bit_AI.market_sell('BTC', decision['amount'])
            if json.loads(sell_sign)['result'] != 'success':
                print("Error Occured: ", end='')
                print(json.loads(sell_sign)['error_code'])
        elif decision['decision'] == 'hold':
            print("Hold decision proceeded")
            pass
    except Exception as e:
        print('Something went wrong.\n Error message: ', e)
        timeout = 30 * 60
        asyncio.run(bit_AI.tel(f"""==== Transaction Recipt ====\n
Decision:\t{decision['decision']}\n
Reason:\t{decision['reason']}\n
Amount:\t{decision['amount']}\n
profit:\t{decision['profit']}\n
Estimated Time:\t{decision['ET']}"""))

    print("======== Transaction Recipt ========")
    print("Decision:\t",        decision['decision'])
    print("Reason:\t",          decision['reason'])
    print("Amount:\t",          decision['amount'])
    print('profit:\t',          decision['profit'])
    print('Estimated Time:\t',  decision['ET'])

    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(bit_AI.tel(f"""==== Transaction Recipt ====\n
Decision:\t{decision['decision']}\n
Reason:\t{decision['reason']}\n
Amount:\t{decision['amount']}\n
profit:\t{decision['profit']}\n
Estimated Time:\t{decision['ET']}"""))

    timeout = 0
    e_time = decision['ET'].split(' ')
    for i in range(len(e_time)):
        if e_time[i][-1] == 'h':
            timeout += int(e_time[i][:-1]) * 60 * 60
        elif e_time[i][-1] == 'm':
            timeout += int(e_time[i][:-1]) * 60
    
    print(timeout)

    bit_AI.write_chat(json.dumps(decision))

    try:
        u_input = inputimeout("Would you like to stop?: ", timeout=timeout)
    except Exception:
        u_input = ''
