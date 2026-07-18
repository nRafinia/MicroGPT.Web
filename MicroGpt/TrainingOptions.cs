namespace MicroGpt;

/// <summary>Optimizer and schedule hyperparameters.</summary>
public sealed record TrainingOptions
{
    /// <summary>Total optimization steps (one document per step).</summary>
    public int Steps { get; init; } = 1000;

    /// <summary>Peak learning rate; decays linearly to 0 over <see cref="Steps"/>.</summary>
    public double LearningRate { get; init; } = 0.01;

    public double Beta1 { get; init; } = 0.85;
    public double Beta2 { get; init; } = 0.99;
    public double Epsilon { get; init; } = 1e-8;

    public void Validate()
    {
        if (Steps < 1) throw new ArgumentOutOfRangeException(nameof(Steps));
        if (LearningRate <= 0) throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (Beta1 is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(Beta1));
        if (Beta2 is < 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(Beta2));
    }
}
