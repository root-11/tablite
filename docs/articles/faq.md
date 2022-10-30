# Frequently Asked Questions

| Question | Answer |
|:---|:---|
|I'm not in a notebook. Is there a nice way to view tables?| Yes. `table.show()` prints the ascii version |
| I'm looking for the equivalent to `apply` in pandas. | Just use list comprehensions: <br>`table[column] = [f(x) for x in table[column]` |
| What about `map`? | Just use the python function:<br> `mapping = map(f, table[column name])` |
| Is there a `where` function? | It's called `any` or `all` like in python:<br> `table.any(column_name > 0)`. |
| I like sql and sqlite. Can I use sql? | Yes. Call `table.to_sql()` returns ANSI-92 SQL compliant table definition.<br>You can use this in any SQL compliant engine. |
| sometimes i need to clean up data with datetimes. Is there any tool to help with that? | Yes. Look at DataTypes.<br>`DataTypes.round(value, multiple)` allows rounding of datetime.