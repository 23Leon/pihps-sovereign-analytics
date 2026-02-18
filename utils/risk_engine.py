def classify_risk(change, volatility):

    score = (change * 50) + (volatility * 50)

    if score > 7:
        level = "HIGH"
    elif score > 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level
