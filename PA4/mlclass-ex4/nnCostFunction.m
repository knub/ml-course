function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% PREDICTION
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
A2 = sigmoid(X * Theta1');
A2 = [ones(m, 1) A2];
A3 = sigmoid(Theta2 * A2')';
% ENDPREDICTION
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
	yi = zeros(num_labels, 1);
	yi(y(i)) = 1;
	for k = 1:num_labels
		J = J + (- yi(k) * log(A3(i, k)) - (1 - yi(k)) * log(1 - (A3(i, k))));
	endfor
endfor

J = J / m;

Theta1New = Theta1(:,2:end);
Theta2New = Theta2(:,2:end);

J = J + lambda * (sum(sum(Theta1New .* Theta1New)) + sum(sum(Theta2New .* Theta2New))) / (2 * m);


% ====================== YOUR CODE HERE ======================

for i = 1:m
	% Step 1
	a1 = X(i,:)';

	%Theta1 = 25 * 401;
	% a1 = 401 * 1;
	% z2 = 25 * 1;
	% Theta2 = 10 * 26;
	% z3 = 10 * 1;
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	% Step 2
	yi = zeros(num_labels, 1);
	yi(y(i)) = 1;
	delta3 = a3 - yi;
	delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2);

	Theta2_grad = Theta2_grad + delta3 * a2';
	Theta1_grad = Theta1_grad + delta2 * a1';
endfor

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
