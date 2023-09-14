// var bytes = [196, 132];
var bytes = [240, 159, 152, 173]
var codePoints = [];
var bele = "le";
var ret = "";
var base = "hex"
var space = true

for (var i = 0; i < bytes.length;) {
    var byte = bytes[i];
    if ((byte & (0b11110000)) == (0b11110000)) {
        var byte4 = bytes[i];
        var byte3 = bytes[i + 1];
        var byte2 = bytes[i + 2];
        var byte1 = bytes[i + 3];
        var codePoint = ((byte4 & (0b00000111)) << 18) | ((byte3 & (0b00111111)) << 12) | ((byte2 & (0b00111111)) << 6) | (byte1 & (0b00111111));
        i += 4;
    } else if ((byte & (0b11100000)) == (0b11100000)) {
        var byte3 = bytes[i];
        var byte2 = bytes[i + 1];
        var byte1 = bytes[i + 2];
        var codePoint = ((byte3 & (0b00001111)) << 12) | ((byte2 & (0b00111111)) << 6) | (byte1 & (0b00111111));
        i += 3;
    } else if ((byte & (0b11000000)) == (0b11000000)) {
        var byte2 = bytes[i];
        var byte1 = bytes[i + 1];
        var codePoint = ((byte2 & (0b00011111)) << 6) | (byte1 & (0b00111111));
        i += 2;
    } else if ((byte & 0x80) == 0) {
        var codePoint = byte;
        i += 1;
    } else {
        throw new Error("Unrecognized byte at position " + (i + 1));
    }
    codePoints.push(codePoint);
}

function codePointToUtf32(codePoint, byteOrder) {
    var b4 = (codePoint & 0xff000000) >> 24;
    var b3 = (codePoint & 0x00ff0000) >> 16;
    var b2 = (codePoint & 0x0000ff00) >> 8;
    var b1 = (codePoint & 0x000000ff);
    if (byteOrder == 'be') {
        return [b4, b3, b2, b1];
    } else if (byteOrder == 'le') {
        return [b1, b2, b3, b4];
    } else {
        throw new Error("Unrecognized byte order request for conversion.");
    }
}

var gucciUtf32 = [];
for (var i = 0; i < codePoints.length; i++) {
    var codePoint = codePoints[i];
    var utf32 = codePointToUtf32(codePoint, bele);
    gucciUtf32.push(utf32);
}


for (var i = 0; i < gucciUtf32.length; i++) {
    var utf32 = gucciUtf32[i];
    var val = (1 << 24) * utf32[0] + (1 << 16) * utf32[1] + (1 << 8) * utf32[2] + utf32[3];
    if (base == 'binary') {
        var binVal = val.toString(2);
        var padLen = 32;
        while (binVal.length < padLen) {
            binVal = "0" + binVal;
        }
        ret += binVal;
    } else if (base == 'octal') {
        var octVal = val.toString(8);
        ret += octVal;
    } else if (base == 'decimal') {
        var decVal = val.toString(10);
        ret += decVal;
    } else if (base == 'hex') {
        var hexVal = val.toString(16);
        var padLen = 8;
        while (hexVal.length < padLen) {
            hexVal = "0" + hexVal;
        }
        ret += hexVal;
    } else if (base == 'raw') {
        var char4 = utf32[0];
        var char3 = utf32[1];
        var char2 = utf32[2];
        var char1 = utf32[3];
        var str = String.fromCharCode(char4) + String.fromCharCode(char3) + String.fromCharCode(char2) + String.fromCharCode(char1);
        ret += str;
    }
    if (space) {
        if (i != gucciUtf32.length - 1) {
            ret += " ";
        }
    }
}


console.log(ret);