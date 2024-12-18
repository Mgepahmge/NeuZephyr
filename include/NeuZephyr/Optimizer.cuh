/**
 * @file Optimizer.cuh
 * @brief Definition of optimization algorithms for training deep learning models.
 *
 * This file declares a set of optimization algorithms designed to update the parameters
 * of deep learning models during training. The algorithms aim to minimize the loss function
 * using different strategies, such as adjusting learning rates dynamically or incorporating
 * momentum to accelerate convergence. Each optimizer implements a `step` function to perform
 * parameter updates, leveraging GPU-based tensor operations for efficiency.
 *
 * @details
 * The optimizers included in this file are:
 * - **SGD (Stochastic Gradient Descent)**: Updates parameters using the negative gradient and a fixed learning rate.
 * - **Momentum**: Improves SGD by adding a momentum term to help smooth out updates and accelerate convergence.
 * - **AdaGrad**: Adjusts learning rates based on historical gradient information, adapting to each parameter's behavior.
 * - **RMSprop**: Uses a moving average of squared gradients to stabilize learning rate adjustments.
 * - **Adam (Adaptive Moment Estimation)**: Combines momentum and RMSprop by computing adaptive learning rates using first and second moment estimates.
 * - **NAdam (Nesterov-accelerated Adam)**: Enhances Adam with Nesterov momentum for improved convergence speed.
 * - **AdaDelta**: An extension of AdaGrad that maintains a running average of squared updates for consistent learning rates.
 *
 * These optimizers are part of the `nz::opt` namespace and are designed for
 * extensibility and high performance in deep learning workflows.
 *
 * @note
 * The `step` function in each optimizer updates model parameters, which are represented
 * by objects compatible with the project's computational graph (e.g., nodes in the graph).
 * Ensure proper memory management and error handling when integrating these optimizers
 * into training processes.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/12/07
 */


#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH
#include <unordered_map>
#include "Nodes.cuh"

/**
 * @namespace nz::opt
 * @brief Contains optimization algorithms for training deep learning models.
 *
 * The `nz::opt` namespace includes a collection of optimization algorithms
 * designed to update model parameters during the training of deep learning models. These
 * optimizers aim to minimize the loss function by adjusting the learning rate dynamically
 * or incorporating momentum terms to improve convergence.
 *
 * @details
 * Key components in this namespace:
 * - **SGD (Stochastic Gradient Descent)**: A basic optimization method that updates model
 *   parameters in the direction of the negative gradient, with a fixed learning rate.
 * - **Momentum**: Enhances SGD by introducing a momentum term that helps accelerate
 *   convergence and reduces oscillations.
 * - **AdaGrad**: An optimizer that adjusts learning rates based on the historical gradients,
 *   allowing it to handle sparse data more effectively.
 * - **RMSprop**: A modification of AdaGrad that uses a moving average of squared gradients
 *   to stabilize the learning rate, leading to more consistent updates.
 * - **Adam (Adaptive Moment Estimation)**: Combines the benefits of momentum and RMSprop,
 *   providing adaptive learning rates for each parameter using first and second moment estimates.
 * - **NAdam (Nesterov-accelerated Adam)**: An improvement over Adam by incorporating Nesterov
 *   momentum, which helps achieve faster convergence.
 * - **AdaDelta**: A variant of AdaGrad that maintains a constant learning rate by using
 *   a running average of squared updates, avoiding the diminishing learning rate problem.
 *
 * These optimizers are designed to work efficiently in high-performance computing environments,
 * utilizing GPU-based tensor operations to accelerate training. The algorithms in this namespace
 * can be easily extended to support additional optimization strategies in the future.
 *
 * This namespace plays a critical role in the optimization of deep learning models by providing
 * a set of tools to adaptively adjust model parameters during training, improving the overall
 * performance and stability of the training process.
 *
 * @note
 * The optimizers in this namespace rely on tensor-based operations for efficient computation.
 * Ensure that proper memory management and error handling are applied when using these algorithms.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/12/07
 */
namespace nz::opt {
    using namespace data;
    using namespace nodes;

    /**
     * @class Optimizer
     * @brief Base class for optimization algorithms in deep learning.
     *
     * The `Optimizer` class serves as the base class for all optimization algorithms used in
     * training deep learning models. It defines the common interface that all optimizer
     * classes must implement, including the `step` function, which updates the model parameters
     * (or nodes) during training based on the optimizer's specific strategy.
     *
     * @details
     * The `Optimizer` class contains a protected member:
     * - **learning_rate**: A scalar value representing the learning rate used in parameter updates.
     *
     * This class is intended to be subclassed by various optimization algorithms, such as
     * SGD, Adam, and AdaGrad. Each subclass is required to implement the `step` function,
     * which is responsible for updating the model parameters according to the specific optimization
     * method being used.
     *
     * @note
     * - The `step` function should be called after calculating the gradients for a given input.
     * - Subclasses should ensure that they implement parameter-specific update logic within
     *   the `step` function.
     * - The `learning_rate` is typically set during the initialization of an optimizer and
     *   is used to control the size of the updates applied to model parameters.
     *
     * This class is part of the `nz::opt` namespace and provides a common
     * structure for implementing various optimizers, facilitating extensibility and code reuse.
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API Optimizer {
    protected:
        Tensor::value_type learning_rate;

    public:
        /**
         * @brief Default constructor for the Optimizer class.
         *
         * This is the default constructor for the `Optimizer` class. It initializes the base
         * class and sets the `learning_rate` to its default value. This constructor does not
         * perform any specific initialization, as it is intended to be used in subclasses where
         * additional initialization might occur.
         *
         * @note
         * - This constructor should typically not be used directly; rather, the derived classes
         *   should be used to initialize specific optimizer instances.
         * - The `learning_rate` is intended to be set by the derived classes during their initialization.
         *
         * @see
         * `Optimizer` for the base class and other methods.
         */
        explicit Optimizer() = default;

        /**
         * @brief Default destructor for the Optimizer class.
         *
         * This is the default destructor for the `Optimizer` class. It ensures proper cleanup
         * of any resources acquired by the class. Since this is a base class, the destructor
         * is virtual to ensure that the destructors of derived classes are called correctly
         * when an object is deleted through a base class pointer.
         *
         * @note
         * - This destructor does not perform any specific cleanup, as the optimizer class does
         *   not manage resources directly. However, derived classes that manage dynamic memory
         *   or other resources should implement their own destructor to handle cleanup appropriately.
         * - The use of a virtual destructor ensures proper resource deallocation in case of polymorphic
         *   object deletion.
         *
         * @see
         * `Optimizer` for the base class and other methods.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        virtual ~Optimizer() = default;

        /**
         * @brief Pure virtual function for performing a single optimization step.
         *
         * This is a pure virtual function that must be overridden by derived optimizer classes.
         * The `step` function is responsible for updating the model parameters (or nodes) based
         * on the optimization algorithm's rules. It takes a `Node` pointer as input, representing
         * the parameters that will be modified in the optimization process.
         *
         * The implementation of this function varies depending on the specific optimization
         * algorithm (e.g., SGD, Adam, Momentum, etc.), but the common goal is to update the
         * model parameters in the direction that minimizes the loss function.
         *
         * @param input A pointer to a `Node` object representing the model's parameters that will
         *              be updated during the optimization step.
         *
         * @note
         * - Since this is a pure virtual function, it must be implemented by all derived classes
         *   of `Optimizer` for the optimization process to work.
         * - The `Node` class is expected to represent model parameters and should support necessary
         *   operations for optimization, such as gradient updates.
         *
         * @see
         * Derived classes like `SGD`, `Adam`, `Momentum`, etc., for specific implementations of this method.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        virtual void step(Node* input) = 0;
    };

    /**
     * @class SGD
     * @brief Stochastic Gradient Descent (SGD) optimizer for deep learning models.
     *
     * The `SGD` class implements the Stochastic Gradient Descent optimization algorithm,
     * which is one of the most basic and widely-used methods for optimizing deep learning
     * model parameters. The algorithm updates the model's parameters by moving in the direction
     * of the negative gradient scaled by a learning rate.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation
     * of the `step` method, which updates the parameters of the model (represented as `Node` objects)
     * using the SGD algorithm.
     *
     * @details
     * - The primary function of this optimizer is to adjust model parameters based on the gradients
     *   and a fixed learning rate. It performs updates to minimize the loss function during training.
     * - The optimizer uses parallel processing on the GPU through CUDA to accelerate the parameter
     *   update process, making it suitable for training large models with many parameters.
     * - While simple, SGD is effective for many machine learning tasks and serves as a foundation
     *   for more advanced optimizers such as Adam and RMSprop.
     * - This optimizer works by updating the weights in the direction that reduces the loss, with
     *   the magnitude of the update controlled by the learning rate.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects,
     *   and these nodes must have associated gradients for the optimizer to function correctly.
     * - It is specifically designed to work with deep learning frameworks that leverage GPU
     *   acceleration for efficient computation.
     *
     * ### Usage Example:
     * ```cpp
     * SGD optimizer(0.01);
     * graph.update(&optimizer) // Suppose "graph" is a computation graph waiting for gradient updates;
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API SGD : public Optimizer {
    public:
        /**
         * @brief Constructor for the SGD optimizer.
         *
         * This constructor initializes the `SGD` optimizer with a specified learning rate.
         * The learning rate is a crucial hyperparameter that determines the step size for each
         * parameter update during training. A smaller learning rate leads to smaller updates, while
         * a larger learning rate results in faster convergence but may risk overshooting the optimal solution.
         *
         * @param learning_rate The learning rate to be used in the optimization process.
         *                      It defines the magnitude of the updates to the model parameters.
         *
         * @note
         * - The learning rate should be chosen carefully, as it significantly impacts the model's
         *   convergence during training. A value that is too large may cause the optimization to diverge,
         *   while a value that is too small may lead to slow convergence.
         *
         * @see SGD for the optimizer class that uses this constructor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit SGD(Tensor::value_type learning_rate);

        /**
         * @brief Performs a single step of the Stochastic Gradient Descent (SGD) optimization.
         *
         * This method updates the model parameters (represented by `Node` objects) using the Stochastic Gradient Descent algorithm.
         * The parameters are updated based on the gradients computed during the backward pass, and the updates are scaled
         * by the learning rate. The method uses CUDA to parallelize the parameter updates on the GPU, ensuring high performance
         * for large-scale models.
         *
         * The update process involves computing the negative gradient and scaling it by the learning rate to adjust the model parameters.
         * This method is intended to be called during the training loop to update the parameters at each iteration.
         *
         * @param input The `Node` object that holds the model parameters and their gradients. This node must have a valid gradient
         *              computed during the backward pass.
         *
         * @note
         * - The method assumes that the `input` node contains a valid `output` tensor with computed gradients.
         * - The computation is performed on the GPU using CUDA, so a CUDA-compatible environment is required.
         * - Ensure that the model parameters have been properly initialized and gradients are computed before calling this method.
         *
         * @see SGD for the class that defines this method.
         * @see Nodes::Node for the class representing the model parameters.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class Momentum
     * @brief Momentum optimizer for deep learning models.
     *
     * The `Momentum` class implements the Momentum optimization algorithm, which is a variant of the Stochastic Gradient Descent (SGD).
     * Momentum helps accelerate SGD in the relevant direction and dampens oscillations, improving convergence speed and stability.
     * It achieves this by incorporating a velocity term that accumulates a fraction of the previous gradients, which is used to update
     * the model parameters in the direction of the accumulated gradients.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters (represented as `Node` objects) using the Momentum algorithm.
     *
     * @details
     * - The optimizer maintains a `velocity` map, which tracks the velocity (accumulated gradients) for each model parameter (`Node`).
     * - The velocity is updated using the formula:
     *   \[
     *   v_{t+1} = \beta v_t + (1 - \beta) g_t
     *   \]
     *   where \( v_t \) is the velocity, \( \beta \) is the momentum factor, and \( g_t \) is the current gradient.
     * - The updated velocity is then used to adjust the model parameters using a learning rate, similar to SGD.
     * - The optimizer uses GPU-accelerated computations through CUDA to efficiently update parameters, making it suitable for large-scale models.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects, and each node must have associated gradients.
     * - The velocity is stored per `Node` object, and if a `Node` does not have an existing velocity, it is initialized to a zero tensor.
     * - The optimizer utilizes GPU memory for velocity storage and gradient computation, requiring CUDA support.
     * - Ensure that the model parameters have been properly initialized, and gradients are computed before calling this method.
     *
     * ### Usage Example:
     * ```cpp
     * Momentum optimizer(0.01, 0.9);
     * graph.update(&optimizer); // Suppose "graph" is a computation graph waiting for gradient updates;
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     * @see Nodes::Node for the class representing model parameters.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API Momentum : public Optimizer {
        std::pmr::unordered_map<Node*, Tensor> velocity;
        Tensor::value_type beta;

    public:
        /**
         * @brief Constructs a Momentum optimizer with a specified learning rate and momentum factor.
         *
         * This constructor initializes a `Momentum` optimizer with a given learning rate and momentum factor.
         * The learning rate controls the step size in the gradient descent update, while the momentum factor
         * helps accelerate the optimizer by incorporating previous gradients.
         *
         * @param learning_rate The learning rate for the optimizer, which determines the step size for parameter updates.
         * @param beta The momentum factor, which controls the influence of previous gradients on the current update.
         *             Typically a value between 0.0 and 1.0, where a value closer to 1 means more influence from previous gradients.
         *
         * @note
         * - The learning rate and momentum factor should be chosen based on the specific task and model being trained.
         * - The optimizer assumes that the model parameters are represented as `Node` objects and that these nodes
         *   will have gradients available when the `step` method is called.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit Momentum(Tensor::value_type learning_rate, Tensor::value_type beta);

        /**
         * @brief Performs a single optimization step using the Momentum algorithm.
         *
         * The `step` function updates the model parameters represented by the `Node` object using the Momentum
         * optimization algorithm. It incorporates both the current gradients and the previous velocity term to
         * update the model parameters. The momentum term helps accelerate the convergence of the optimizer by
         * smoothing out updates and reducing oscillations.
         *
         * This method performs the following steps:
         * - Initializes the velocity vector for the `Node` if it is not already available. The velocity vector
         *   stores the running average of past gradients, scaled by the momentum factor.
         * - Allocates memory for temporary variables on the GPU and computes the velocity update using a CUDA kernel.
         * - Updates the velocity vector and the model parameters by applying the momentum update and gradient descent.
         * - Frees the temporary GPU memory after the update is complete.
         *
         * @param input A pointer to the `Node` object representing the model parameters. This object should have
         *              gradients stored in its `output` attribute, which will be used to update the parameters.
         *
         * @note
         * - The `Node` object is assumed to have a valid `output` tensor with its gradients already computed.
         * - The velocity map stores the velocity for each `Node` to ensure the momentum is correctly applied per parameter.
         * - The method leverages CUDA to perform parallel computations for efficiency during the optimization process.
         * - The optimizer uses the momentum factor (`beta`) to control the influence of past gradients on the current update.
         *
         * @see Momentum for the class definition and constructor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class AdaGrad
     * @brief AdaGrad optimizer for deep learning models.
     *
     * The `AdaGrad` class implements the Adaptive Gradient algorithm, which is a popular optimization method
     * that adapts the learning rate for each parameter based on the historical gradients. AdaGrad is known for
     * its ability to handle sparse gradients and adjust learning rates during training.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters using the AdaGrad algorithm.
     *
     * @details
     * - The main idea of AdaGrad is to maintain a separate learning rate for each parameter by scaling the gradient
     *   based on the sum of squares of past gradients. This helps reduce the learning rate for frequently updated
     *   parameters and increases it for rarely updated ones.
     * - AdaGrad can significantly improve training performance for problems with sparse data or parameters that
     *   have widely varying scales.
     * - This optimizer is effective for tasks such as natural language processing or training deep learning models
     *   with sparse gradients.
     * - The optimizer uses parallel GPU processing with CUDA to speed up parameter updates, especially when dealing
     *   with large models.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects, and these nodes must have
     *   associated gradients for the optimizer to function correctly.
     * - The `gss` map stores the sum of squared gradients for each parameter, which is used to adjust the learning rate.
     * - The `epsilon` term ensures numerical stability when dividing by the sum of squared gradients.
     *
     * ### Usage Example:
     * ```cpp
     * AdaGrad optimizer(0.01);
     * graph.update(&optimizer) // Suppose "graph" is a computation graph waiting for gradient updates;
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API AdaGrad : public Optimizer {
        std::unordered_map<Node*, Tensor> gss;
        Tensor::value_type epsilon = 1e-6;

    public:
        /**
         * @brief Constructs an AdaGrad optimizer with the specified learning rate.
         *
         * This constructor initializes the `AdaGrad` optimizer with the given learning rate, which is used to control
         * the magnitude of the updates during training. The learning rate determines how much to adjust the model's
         * parameters in response to the computed gradients.
         *
         * @param learning_rate The learning rate to be used for parameter updates. It is a scalar value that controls
         *                      the size of the steps taken during the optimization process. A smaller value makes the
         *                      updates more conservative, while a larger value can speed up convergence but may cause
         *                      instability.
         *
         * @note
         * - The `epsilon` value used in the AdaGrad algorithm is set to a default of `1e-6` for numerical stability
         *   during updates and is not modified by this constructor.
         * - The optimizer assumes that the model parameters are represented by `Node` objects, and the gradients for
         *   these nodes will be updated during the `step` method.
         *
         * @see AdaGrad for the full class definition.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit AdaGrad(Tensor::value_type learning_rate);

        /**
         * @brief Performs a single optimization step using the AdaGrad algorithm.
         *
         * The `step` function updates the model parameters represented by the `Node` object using the AdaGrad
         * optimization algorithm. AdaGrad adapts the learning rate for each parameter by considering the history
         * of gradients, providing faster convergence for sparse gradients.
         *
         * This method performs the following steps:
         * - Initializes the sum of squared gradients (GSS) for the parameter (`Node`) if it has not been initialized.
         * - Allocates memory on the GPU for storing intermediate results and computes the AdaGrad update for the model parameters.
         * - Uses the sum of squared gradients to scale the gradient and update the model parameters.
         * - Frees the temporary memory allocated for computations after the update.
         *
         * @param input A pointer to the `Node` object representing the model parameters. This object should have
         *              gradients stored in its `output` attribute, which will be used to update the parameters.
         *
         * @note
         * - The `Node` object is assumed to have a valid `output` tensor with its gradients already computed.
         * - The `gss` map stores the sum of squared gradients for each parameter, ensuring that the learning rate
         *   adapts to the frequency of gradient updates.
         * - The `epsilon` term is used to avoid division by zero and ensure numerical stability when updating the parameters.
         * - The method leverages CUDA for parallel computation, which speeds up the update process, especially for large models.
         *
         * @see AdaGrad for the class definition and constructor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class RMSprop
     * @brief RMSprop optimizer for deep learning models.
     *
     * The `RMSprop` class implements the RMSprop (Root Mean Square Propagation) optimization algorithm.
     * RMSprop is designed to address the diminishing learning rate issue of AdaGrad by introducing a
     * moving average of squared gradients. This helps stabilize the learning rate, making it suitable
     * for non-stationary or dynamically changing loss functions.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters (represented as `Node` objects) using the RMSprop algorithm.
     *
     * @details
     * - RMSprop maintains an exponentially decaying average of squared gradients for each parameter.
     * - The learning rate is adjusted based on this average, which helps prevent the learning rate from decaying too quickly.
     * - The update rule for RMSprop can be expressed as:
     *   \f[
     *   v_t = \beta v_{t-1} + (1 - \beta) g_t^2
     *   \f]
     *   \f[
     *   \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t
     *   \f]
     *   where:
     *   - \( v_t \) is the moving average of squared gradients.
     *   - \( \beta \) is the decay rate (usually between 0.9 and 0.99).
     *   - \( g_t \) is the current gradient.
     *   - \( \eta \) is the learning rate.
     *   - \( \epsilon \) is a small value to ensure numerical stability.
     *
     * - RMSprop is widely used in training recurrent neural networks (RNNs) and other deep learning models
     *   where the loss function can change dynamically.
     *
     * @note
     * - The optimizer assumes that model parameters are represented by `Node` objects, and these nodes
     *   have gradients computed before calling the `step` method.
     * - The `v` map stores the moving average of squared gradients for each parameter.
     * - The `epsilon` term helps avoid division by zero and ensures numerical stability.
     *
     * ### Usage Example:
     * ```cpp
     * RMSprop optimizer(0.001, 0.9);
     * graph.update(&optimizer); // Suppose "graph" is a computation graph waiting for gradient updates.
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API RMSprop : public Optimizer {
        std::unordered_map<Node*, Tensor> v;
        Tensor::value_type decay_rate;
        Tensor::value_type epsilon = 1e-6;

    public:
        /**
         * @brief Constructs an RMSprop optimizer with specified learning rate and decay rate.
         *
         * The constructor initializes the RMSprop optimizer with the provided learning rate and decay rate.
         * RMSprop is an adaptive learning rate optimization algorithm that maintains a moving average of squared gradients
         * to scale the learning rate for each parameter individually.
         *
         * This constructor sets the initial values for:
         * - `learning_rate`: The step size used for parameter updates.
         * - `decay_rate`: The factor used to update the moving average of squared gradients, which controls how much
         *   the previous gradients influence the current update.
         *
         * @param learning_rate The learning rate used in the RMSprop algorithm to scale the gradient updates.
         * @param decay_rate The decay rate (also called momentum term) used to compute the moving average of squared gradients.
         *                   A higher value gives more weight to previous gradients, while a lower value emphasizes recent gradients.
         *
         * @note
         * - The default value of epsilon (`1e-6`) is used to avoid division by zero during parameter updates.
         * - The decay rate should typically be a value between 0.9 and 0.99, with a default value of 0.9 commonly used in practice.
         * - This constructor ensures that the optimizer is properly initialized with the necessary hyperparameters before calling
         *   the `step` method to perform optimization steps.
         *
         * @see RMSprop for the full class definition, and `step` for the optimization step implementation.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit RMSprop(Tensor::value_type learning_rate, Tensor::value_type decay_rate);

        /**
         * @brief Performs a single optimization step using the RMSprop algorithm.
         *
         * The `step` method updates the model parameters based on the gradients computed
         * during the forward pass. It applies the RMSprop optimization algorithm, which uses
         * a moving average of the squared gradients to adjust the learning rate for each parameter.
         * This helps to maintain a stable and adaptive learning rate, preventing the gradient
         * from becoming too large or too small during training.
         *
         * The method checks if the squared gradient cache (`v`) for the given input node exists.
         * If not, it initializes it to zero. Then, it applies the RMSprop update rule using the
         * current gradient, the moving average of squared gradients, and the specified learning rate
         * and decay rate.
         *
         * This method is designed to be used with a model parameter represented as a `Node` object
         * and assumes that the node has an associated output and gradient.
         *
         * @param input A pointer to the `Node` object representing the model parameter to be updated.
         *              The node should have an output tensor and its gradient already computed.
         *
         * @note
         * - This method operates on the GPU using CUDA to accelerate the parameter update process.
         * - It assumes that the `input` node has a valid gradient stored in its `output` object.
         * - The squared gradient cache (`v`) is maintained for each node individually.
         *
         * @see RMSprop for the class definition and constructor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class Adam
     * @brief Adam optimizer for deep learning models.
     *
     * The `Adam` class implements the Adam optimization algorithm, which is an adaptive learning rate optimization method
     * designed for training deep learning models. Adam combines the advantages of two popular optimization techniques:
     * Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
     * It uses estimates of first and second moments of gradients to adaptively adjust the learning rate for each parameter.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters (represented as `Node` objects) using the Adam algorithm.
     *
     * @details
     * - The optimizer maintains two moment estimates for each parameter (`Node`):
     *   - \( m_t \): The first moment estimate, which is the exponentially decaying average of past gradients.
     *   - \( v_t \): The second moment estimate, which is the exponentially decaying average of past squared gradients.
     * - The moment estimates are updated using the following formulas:
     *   \[
     *   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
     *   \]
     *   \[
     *   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
     *   \]
     *   where \( g_t \) is the current gradient, \( \beta_1 \) and \( \beta_2 \) are the decay rates for the first and second moments.
     * - The model parameters are then updated using the bias-corrected moment estimates:
     *   \[
     *   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
     *   \]
     *   \[
     *   \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
     *   \]
     *   where \( \eta \) is the learning rate, \( \epsilon \) is a small constant to prevent division by zero.
     * - The optimizer uses GPU-accelerated computations through CUDA to efficiently update parameters, making it suitable for large-scale models.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects, and each node must have associated gradients.
     * - The first and second moment estimates (`m` and `v`) are stored per `Node` object. If a `Node` does not have existing moments,
     *   they are initialized to zero tensors.
     * - The optimizer utilizes GPU memory for moment storage and gradient computation, requiring CUDA support.
     * - Ensure that the model parameters have been properly initialized, and gradients are computed before calling this method.
     *
     * ### Usage Example:
     * ```cpp
     * Adam optimizer(0.001, 0.9, 0.999);
     * graph.update(&optimizer); // Suppose "graph" is a computation graph waiting for gradient updates.
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     * @see Nodes::Node for the class representing model parameters.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API Adam : public Optimizer {
        std::unordered_map<Node*, Tensor> m;
        std::unordered_map<Node*, Tensor> v;
        Tensor::value_type beta1;
        Tensor::value_type beta2;
        int it;
        Tensor::value_type epsilon = 1e-6;

    public:
        /**
         * @brief Constructs an Adam optimizer with the specified hyperparameters.
         *
         * The `Adam` constructor initializes an instance of the Adam optimizer with the given learning rate,
         * beta1, and beta2 values. These hyperparameters control the behavior of the Adam optimization algorithm:
         * - The learning rate determines the step size for parameter updates.
         * - Beta1 controls the decay rate for the first moment estimate (moving average of gradients).
         * - Beta2 controls the decay rate for the second moment estimate (moving average of squared gradients).
         *
         * The constructor also initializes the internal iteration counter (`it`) to zero, which is used for bias correction
         * during the parameter updates.
         *
         * @param learning_rate The learning rate (\( \eta \)) used for parameter updates. It controls the step size.
         * @param beta1 The exponential decay rate for the first moment estimate (\( \beta_1 \)).
         *              Typical values are in the range [0.9, 0.99].
         * @param beta2 The exponential decay rate for the second moment estimate (\( \beta_2 \)).
         *              Typical values are in the range [0.99, 0.999].
         *
         * @note
         * - The learning rate, beta1, and beta2 values should be chosen carefully based on the specific task and dataset.
         * - The default values for beta1 (0.9) and beta2 (0.999) are commonly used in practice.
         *
         * @see Adam::step for the method that performs parameter updates using the Adam optimizer.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit Adam(Tensor::value_type learning_rate, Tensor::value_type beta1, Tensor::value_type beta2);

        /**
         * @brief Performs a single optimization step using the Adam algorithm.
         *
         * The `step` method updates the model parameters based on the gradients computed during the forward pass.
         * It applies the Adam optimization algorithm, which uses moving averages of the gradients and their squared values
         * to adaptively adjust the learning rate for each parameter. This helps achieve stable and efficient parameter updates.
         *
         * The method performs the following steps:
         * 1. Increments the internal iteration counter (`it`), which is used for bias correction.
         * 2. Checks if the first moment estimate (`m`) and second moment estimate (`v`) for the given input node exist.
         *    If not, it initializes them to zero tensors with the same shape as the node's output.
         * 3. Launches a CUDA kernel to compute the Adam updates for the parameters, using the current gradient, the moving averages
         *    of the gradients (`m`), and their squared values (`v`), along with the specified hyperparameters (learning rate, beta1, beta2, epsilon).
         *
         * This method is designed to be used with a model parameter represented as a `Node` object and assumes that the node has
         * an associated output tensor and gradient.
         *
         * @param input A pointer to the `Node` object representing the model parameter to be updated.
         *              The node should have an output tensor and its gradient already computed.
         *
         * @note
         * - This method operates on the GPU using CUDA to accelerate the parameter update process.
         * - It assumes that the `input` node has a valid gradient stored in its `output` object.
         * - The first moment estimate (`m`) and second moment estimate (`v`) are maintained for each node individually.
         * - The `epsilon` value is used to prevent division by zero during the parameter update.
         *
         * @see Adam for the class definition and constructor.
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class NAdam
     * @brief NAdam optimizer for deep learning models.
     *
     * The `NAdam` class implements the Nesterov-accelerated Adaptive Moment Estimation (NAdam) optimization algorithm,
     * which combines the benefits of the Adam optimizer with Nesterov momentum. NAdam improves upon Adam by incorporating
     * Nesterov momentum into the first moment estimation, which can lead to faster convergence in some scenarios.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters (represented as `Node` objects) using the NAdam algorithm.
     *
     * @details
     * - The optimizer maintains three tensors for each parameter (`Node`):
     *   - \( m_t \): The first moment estimate, which is the exponentially decaying average of past gradients.
     *   - \( m_t' \): The modified first moment estimate, incorporating Nesterov momentum.
     *   - \( v_t \): The second moment estimate, which is the exponentially decaying average of past squared gradients.
     * - The moment estimates are updated using the following formulas:
     *   \[
     *   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
     *   \]
     *   \[
     *   m_t' = \beta_1 m_t + (1 - \beta_1) g_t
     *   \]
     *   \[
     *   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
     *   \]
     *   where \( g_t \) is the current gradient, \( \beta_1 \) and \( \beta_2 \) are the decay rates for the first and second moments.
     * - The model parameters are then updated using the bias-corrected moment estimates:
     *   \[
     *   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{m}_t' = \frac{m_t'}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
     *   \]
     *   \[
     *   \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t'}{\sqrt{\hat{v}_t} + \epsilon}
     *   \]
     *   where \( \eta \) is the learning rate, \( \epsilon \) is a small constant to prevent division by zero.
     * - The optimizer uses GPU-accelerated computations through CUDA to efficiently update parameters, making it suitable for large-scale models.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects, and each node must have associated gradients.
     * - The first moment estimate (`m`), modified first moment estimate (`m_modified`), and second moment estimate (`v`) are stored per `Node` object.
     *   If a `Node` does not have existing moments, they are initialized to zero tensors.
     * - The optimizer utilizes GPU memory for moment storage and gradient computation, requiring CUDA support.
     * - Ensure that the model parameters have been properly initialized, and gradients are computed before calling this method.
     *
     * ### Usage Example:
     * ```cpp
     * NAdam optimizer(0.001, 0.9, 0.999);
     * graph.update(&optimizer); // Suppose "graph" is a computation graph waiting for gradient updates.
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     * @see Nodes::Node for the class representing model parameters.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API NAdam : public Optimizer {
        std::unordered_map<Node*, Tensor> m;
        std::unordered_map<Node*, Tensor> m_modified;
        std::unordered_map<Node*, Tensor> v;
        Tensor::value_type beta1;
        Tensor::value_type beta2;
        int it;
        Tensor::value_type epsilon = 1e-6;

    public:
        /**
         * @brief Constructs a NAdam optimizer with specified hyperparameters.
         *
         * Initializes the NAdam (Nesterov-accelerated Adaptive Moment Estimation) optimizer
         * with user-defined learning rate and momentum parameters. NAdam combines the benefits
         * of Nesterov accelerated gradient and Adam optimization techniques, providing
         * adaptive learning rates for each parameter while incorporating momentum.
         *
         * The constructor sets up the initial state of the optimizer, including the learning
         * rate, exponential decay rates for moment estimates, and initializes the iteration
         * counter to zero. This prepares the optimizer for the first optimization step in
         * the training process.
         *
         * @param learning_rate The base learning rate that controls the step size during optimization.
         *                      A smaller value leads to more conservative updates, while a larger
         *                      value allows for more aggressive parameter adjustments.
         *
         * @param beta1 The exponential decay rate for the first moment estimate (moving average
         *              of gradients). Typically set close to 1 (e.g., 0.9) to control the
         *              influence of past gradients on the current update.
         *
         * @param beta2 The exponential decay rate for the second moment estimate (moving average
         *              of squared gradients). Typically set close to 1 (e.g., 0.999) to adapt
         *              the learning rate for each parameter based on its historical gradient information.
         *
         * @note
         * - The iteration counter `it` is initialized to 0, which is critical for the first
         *   bias correction step in the NAdam algorithm.
         * - Recommended default values are learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999.
         * - The hyperparameters significantly impact the optimization process and may require
         *   tuning based on the specific machine learning task.
         *
         * @see Adam, RMSprop Optimization algorithms with similar adaptive learning rate strategies
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit NAdam(Tensor::value_type learning_rate, Tensor::value_type beta1, Tensor::value_type beta2);

        /**
         * @brief Performs a single optimization step using the NAdam algorithm.
         *
         * This method updates the model parameters for a given input node using the Nesterov-accelerated
         * Adaptive Moment Estimation (NAdam) optimization algorithm. It manages the adaptive learning
         * rates and momentum for individual parameters by maintaining and updating first and second
         * moment estimates.
         *
         * The method performs several key operations:
         * 1. Increments the iteration counter
         * 2. Initializes moment and modified moment tensors if they don't exist for the input node
         * 3. Prepares CUDA grid and block configurations for parallel parameter updates
         * 4. Invokes a CUDA kernel to apply the NAdam update rule
         *
         * The initialization of moment tensors ensures that each parameter has its own adaptive
         * learning rate and momentum, allowing for more flexible and efficient optimization across
         * different model parameters.
         *
         * @param input A pointer to the `Node` object representing the model parameter to be updated.
         *              The node must have a valid output tensor and its gradient already computed.
         *
         * @note
         * - This method assumes the input node has a valid gradient stored in its output object.
         * - Moment tensors are created lazily (on-demand) for each unique input node.
         * - The method uses CUDA for parallel computation of parameter updates.
         * - The iteration counter is crucial for bias correction in the NAdam algorithm.
         *
         * @see NAdam::NAdam() Constructor for initializing optimizer parameters
         * @see krnl::NAdam CUDA kernel implementing the NAdam update rule
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };

    /**
     * @class AdaDelta
     * @brief AdaDelta optimizer for deep learning models.
     *
     * The `AdaDelta` class implements the AdaDelta optimization algorithm, which is a variant of the Adagrad optimizer
     * that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients,
     * AdaDelta restricts the window of accumulation to a fixed size, allowing for more robust updates and addressing the diminishing
     * learning rate problem.
     *
     * This class extends the `Optimizer` base class and provides a concrete implementation of the `step` method,
     * which updates the model's parameters (represented as `Node` objects) using the AdaDelta algorithm.
     *
     * @details
     * - The optimizer maintains two accumulators for each parameter (`Node`):
     *   - \( E[g^2]_t \): The exponentially decaying average of past squared gradients.
     *   - \( E[\Delta x^2]_t \): The exponentially decaying average of past squared parameter updates.
     * - The accumulators are updated using the following formulas:
     *   \[
     *   E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2
     *   \]
     *   \[
     *   \Delta x_t = - \frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t
     *   \]
     *   \[
     *   E[\Delta x^2]_t = \rho E[\Delta x^2]_{t-1} + (1 - \rho) \Delta x_t^2
     *   \]
     *   where \( g_t \) is the current gradient, \( \rho \) is the decay rate, and \( \epsilon \) is a small constant to prevent division by zero.
     * - The model parameters are updated using \( \Delta x_t \), which is computed adaptively based on the ratio of the two accumulators.
     * - The optimizer uses GPU-accelerated computations through CUDA to efficiently update parameters, making it suitable for large-scale models.
     *
     * @note
     * - The optimizer assumes that the model parameters are represented by `Node` objects, and each node must have associated gradients.
     * - The accumulators (`acc_grad` and `acc_delta`) are stored per `Node` object. If a `Node` does not have existing accumulators,
     *   they are initialized to zero tensors.
     * - The optimizer utilizes GPU memory for accumulator storage and gradient computation, requiring CUDA support.
     * - Ensure that the model parameters have been properly initialized, and gradients are computed before calling this method.
     *
     * ### Usage Example:
     * ```cpp
     * AdaDelta optimizer(0.95); // rho = 0.95
     * graph.update(&optimizer); // Suppose "graph" is a computation graph waiting for gradient updates.
     * ```
     *
     * @see Optimizer for the base class that defines the interface for all optimizers.
     * @see Nodes::Node for the class representing model parameters.
     *
     * @author
     * Mgepahmge (https://github.com/Mgepahmge)
     *
     * @date
     * 2024/12/07
     */
    class DL_API AdaDelta : public Optimizer {
        std::unordered_map<Node*, Tensor> acc_delta;
        std::unordered_map<Node*, Tensor> acc_grad;
        Tensor::value_type epsilon = 1e-6;

    public:
        /**
         * @brief Constructs an AdaDelta optimizer with a specified decay rate.
         *
         * Initializes the AdaDelta optimization algorithm with a given decay rate (rho).
         * AdaDelta is an adaptive learning rate method that automatically adjusts the
         * learning rate for each parameter, addressing some limitations of traditional
         * stochastic gradient descent methods.
         *
         * Unlike other adaptive optimization algorithms, AdaDelta does not require an
         * explicit learning rate. Instead, it uses a running average of squared gradients
         * and squared parameter updates to scale the optimization step dynamically.
         *
         * @param rho The decay rate that controls the moving window for accumulating
         *            gradient statistics. This parameter determines how quickly the
         *            algorithm forgets past gradient information.
         *            Typically set between 0.9 and 0.999.
         *
         * @note
         * - The `rho` parameter is analogous to the momentum decay rates in other
         *   adaptive optimization algorithms.
         * - A value closer to 1 results in a longer memory of past gradients,
         *   while a value closer to 0 makes the algorithm more responsive to recent gradients.
         * - Default recommended value is often around 0.95.
         *
         * @see RMSprop, Adam Alternative adaptive optimization algorithms
         * @see AdaDelta::step Method that applies the AdaDelta update rule
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        explicit AdaDelta(Tensor::value_type rho);

        /**
         * @brief Performs a single optimization step using the AdaDelta algorithm.
         *
         * This method updates the model parameters for a given input node using the AdaDelta
         * optimization algorithm. It manages adaptive learning rates by maintaining running
         * accumulators for both gradient and parameter update magnitudes.
         *
         * The method performs several key operations:
         * 1. Lazily initializes accumulators for parameter updates and gradients if they don't exist
         * 2. Prepares CUDA grid and block configurations for parallel parameter updates
         * 3. Invokes a CUDA kernel to apply the AdaDelta update rule
         *
         * The lazy initialization of accumulators ensures that each parameter has its own
         * adaptive learning rate, allowing for more flexible and efficient optimization across
         * different model parameters.
         *
         * @param input A pointer to the `Node` object representing the model parameter to be updated.
         *              The node must have a valid output tensor and its gradient already computed.
         *
         * @note
         * - This method assumes the input node has a valid gradient stored in its output object.
         * - Accumulators for parameter updates and gradients are created on-demand for each unique input node.
         * - The method uses CUDA for parallel computation of parameter updates.
         * - The algorithm adapts the learning rate based on the historical gradient information.
         *
         * @see AdaDelta::AdaDelta() Constructor for initializing optimizer parameters
         * @see krnl::AdaDelta CUDA kernel implementing the AdaDelta update rule
         *
         * @author
         * Mgepahmge (https://github.com/Mgepahmge)
         *
         * @date
         * 2024/12/07
         */
        void step(Node* input) override;
    };
}

#endif // OPTIMIZER_CUH
