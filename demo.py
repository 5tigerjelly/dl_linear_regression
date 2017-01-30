from numpy import *

def compute_error_for_line_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) **2
	return totalError / float(len(points))
def gradient_decent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	#starting b and m
	b = starting_b
	m = starting_m

	#gradient descent
	for i in range(num_iterations):
		#update b and m with more accurate b and m
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b,m]

def step_gradient(b_current, m_current, points, learning_rate):

	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		# direction with respect to b and m
		#computing partial derivatives of our error function
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	#update our b and m values using our partial derivatieves
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]




def run():
	points = genfromtxt('data.csv', delimiter=',')

	#step 2
	#how fast should our model converge?
	learning_rate = 0.0001
	#y = mx + b (slope formula)
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	#step3 - train the model
	print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
	[b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print 'ending gradient descent at b = {1}, m = {2}, error = {3}'.format(num_iterations,b,m, compute_error_for_line_given_points(b, m, points))






if __name__ == '__main__':
	run()