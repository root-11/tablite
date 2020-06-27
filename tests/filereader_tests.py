from table import *


def text_escape_tests():
    te = text_escape('"t"')
    assert te == ("t",)

    te = text_escape('"t";"3";"2"')
    assert te == ("t", "3", "2")

    te = text_escape('"this";"123";234;"this";123;"234"')
    assert te == ('this', '123', '234', 'this', '123', '234')

    te = text_escape('"this";"123";"234"')
    assert te == ("this", "123", "234")

    te = text_escape('"this";123;234')
    assert te == ("this", "123", "234")

    te = text_escape('"this";123;"234"')
    assert te == ("this", "123", "234")

    te = text_escape('123;"1\'3";234')
    assert te == ("123", "1'3", "234"), te

    te = text_escape('"1000627";"MOC;SEaert;pás;krk;XL;černá";"2.180,000";"CM3";2')
    assert te == ("1000627", "MOC;SEaert;pás;krk;XL;černá", "2.180,000", "CM3", '2')

    te = text_escape('"1000294";"S2417DG 24"" LED monitor (210-AJWM)";"47.120,000";"CM3";3')
    assert te == ('1000294', 'S2417DG 24"" LED monitor (210-AJWM)', '47.120,000', 'CM3', '3')
