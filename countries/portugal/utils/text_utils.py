import unicodedata


def normalize_municipality_name(name):
    """
    Normalize municipality names by removing diacritical marks and converting to lowercase.
    
    Args:
        name: String or float value to normalize
        
    Returns:
        Normalized string (lowercase, without diacritics) or the original value if float
    """
    if type(name) == float:
        return name
    else:
        # Normalize the string (remove diacritical marks)
        name_without_diacritics = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        # Convert to lowercase
        return name_without_diacritics.lower()
