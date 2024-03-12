import os



if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        if task == 1:
            asyncio.run(process_task_1())
        # elif task == 3: call process_task_3()
        elif task == 2:
            populate_global_resources("./mock_patients")
            asyncio.run(process_task_2())
        elif task == 3: 
            process_task_3()
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
    else:
        print("Please provide a task number as a command-line argument.")