print("========= Welcome to GEMINI BTC TRADER =========")
print("initializing...")
import bit_AI
import time
import warnings
warnings.filterwarnings('ignore')
import datetime
import json
import asyncio
import Prud_AI
import sqlite3 as sql

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

bit_instruction = './Bitcoin Gemini Instruction.md'
bit_chat_session = bit_AI.gen_bit_model(get_instructions(bit_instruction))

former_dec = 3 # 1 = sell/ 0 = buy/ 3 = initial

u_input = ''
timeout= 0
ticker = 0
timestamp = datetime.datetime.now()

while u_input not in ['Y', 'y', 'Yes', 'yes', 'YES']:
    if ticker >= timeout:
        dbop = sql.connect("./Record.db")
        dbcs = dbop.cursor()
        stamp = list(dbcs.execute("SELECT DATE FROM PRUDENCERECORD;"))
        print(stamp)
        # if ((datetime.datetime.now() - timestamp) > datetime.timedelta(hours=23) or len(stamp) == 0 or stamp[-1][0] != str(timestamp)[:10]) and stamp[-1][0] != str(datetime.datetime.now())[:10]:
        # Only for 11-09
        if False:
            print("New day detected. Prudence AI will be processed...")
    
            pru_instruction = './Prudence Gemini Instruction.md'
            pru_chat_session = Prud_AI.gen_pru_model(get_instructions(pru_instruction))
            b = Prud_AI.gem_pru_sug(pru_chat_session)
            bit_AI.model_usage(b)
    
            try:
                with sql.connect("./Record.db") as dbop:
                    dbcs = dbop.cursor()
                    dateRecord = dbcs.execute('SELECT DATE FROM PRUDENCERECORD;')
                    checker = [j for i in list(dateRecord) for j in i]
                    print(len(checker))
                    
                    # initialize prudence record
                    if len(checker) == 11:
                        print("Prudence Record Initialize")
                        for date in checker[:7]:
                            dbcs.execute(f"DELETE FROM PRUDENCERECORD WHERE DATE = '{date}';")
                            print(f"{date} data deleted")
    
                    if str(datetime.datetime.now())[:10] not in checker:
                        dbcs.execute("INSERT INTO PRUDENCERECORD VALUES(?,?,?);", [b[i] for i in list(b.keys())])
                        pass
                    dbop.commit()
            except Exception as e:
                print("Error occured: ", e)
    
            print(b)
            # reset timestamp
            timestamp = datetime.datetime.now()
    
        print("Trading AI will be processed...")
        time.sleep(30)
        try:
            decision = bit_AI.gem_sug(bit_chat_session, bit_AI.get_today_prudence())
            bit_AI.model_usage(decision)
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
            # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(bit_AI.tel(f"ERROR OCCURED:\t{e}"))
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
        print("Decision:\t",        decision)
        # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(bit_AI.tel(f"""==== Transaction Recipt ====\n{decision}"""))
    
        timeout = 0
        e_time = decision['ET'].split(' ')
        for i in range(len(e_time)):
            if e_time[i][-1] == 'h':
                timeout += int(e_time[i][:-1]) * 60 * 60
            elif e_time[i][-1] == 'm':
                timeout += int(e_time[i][:-1]) * 60
        
        print(timeout)
        ticker = 0
        bit_AI.write_chat(json.dumps(decision))
    else:
        ticker += 60
        time.sleep(60)

