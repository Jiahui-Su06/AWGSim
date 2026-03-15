def clamp(
	x: float,
	a: float,
	b: float
) -> float:
	"""Limit x value within [a,b] range.
	
	Args:
        x: The value to clamp.
		a: The lower bound.
		b: The upper bound.
		
	Returns:
        The clamped value.
	
	Raises:
        ValueError: If ``a`` is greater than ``b``.
	"""
	if a > b:
		raise ValueError("a must be <= b")
	return min(max(x,a),b)
