def get_current_weather(arguments):
    location = arguments.get('location')
    return f"It's hot in {location}"


function_map = {
    "joke_of_the_day": joke_of_the_day,
    "get_current_weather": get_current_weather
}

func = function_map.get(function_name)
result = func(arguments)
print(result)