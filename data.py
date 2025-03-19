import random
import pandas as pd
import numpy as np
import constants as const


class DataGenerator:
    @staticmethod
    def generate_data(depth, gestational_age, number_of_participants, trisomy):
        """
        Generate a DataFrame with simulated genomic data for participants.

        Parameters:
        number_of_reads (int): Total number of reads to simulate.
        gestational_age (int): Gestational age in weeks.
        number_of_participants (int): Number of participants to simulate.

        Returns:
        pd.DataFrame: A DataFrame with the simulated data.
        """
        fetal_fraction_mean = const.ff_mean[gestational_age]
        fs_std = const.ff_std[gestational_age]

        half_size = number_of_participants // 2
        number_of_reads = int((depth * const.genome_size) / const.read_length)
        dataframe = pd.DataFrame({
            const.feature_coverage_depth: depth,
            const.feature_age: gestational_age,
            const.feature_n: number_of_reads,
            const.feature_ff: np.maximum(0, np.random.normal(fetal_fraction_mean, fs_std, number_of_participants)),
        })
        trisomy_labels = np.array([1] * half_size + [0] * half_size)
        np.random.shuffle(trisomy_labels)  # Shuffle to randomize the order

        x_values = np.zeros(number_of_participants, dtype=int)
        p = const.get_c(trisomy)
        x_values[trisomy_labels == 0] = np.random.binomial(n=number_of_reads, p=p, size=half_size)

        x_values[trisomy_labels == 1] = np.random.binomial(
            n=number_of_reads,
            p=p * (1 + dataframe.loc[trisomy_labels == 1, const.feature_ff] / 2),
            size=half_size
        )

        dataframe[const.feature_x] = x_values
        dataframe[const.feature_trisomy_label] = trisomy_labels
        dataframe[const.feature_trisomy_type] = trisomy
        return dataframe

    @staticmethod
    def generate_full_dataframe(
                                number_of_participants,
                                coverage_depths,
                                pregnancy_weeks,
                                trisomies):
        data_bases = []
        for trisomy in trisomies:
            for coverage_depth in coverage_depths:
                for week in pregnancy_weeks:
                    data_bases.append(DataGenerator.generate_data(coverage_depth, week, number_of_participants, trisomy))
        full_dataframe = pd.concat(data_bases, ignore_index=True)
        return full_dataframe

if __name__ == '__main__':
    df = DataGenerator.generate_full_dataframe(const.number_of_participants,
                                               const.coverage_depths,
                                               const.pregnancy_weeks,
                                               const.trisomy_types)
    df.to_csv('data.csv')
