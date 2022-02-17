# Optmisation-Jansen-Mecanism

Oscar Fossey (ENSAM), Arthur Lejeune (ENSAM) and Clément Eiserloh (ENSAM).

## Goal 

Find the fastest and robust geometry for a Jansen mecanism.

**Jansen mecanism :**

![alt text](https://github.com/oscarfossey/Optmisation-Jansen-Mecanism/blob/main/Images/Jansen.JPG)

## Motivations

To have the fastest Jansen mecanism there is a known optimal theroric geometry. However this geometry is not robust, which means that with small variations of the length of the bars the has a huge dump in speed. The main issues is that when manufacturing the mecanism the lenght exact length are differents from the nominales length, then there is no assurance that the mecanism will work well and at max speed. Which is the fastest geomatry that is robust?

## Method

We evaluate the robustness with the Monte Carlo algorithm.

For a specific geometry their is a specific robustness score with the help of the simulation module.

To find the best geometry with the best nominales length for the bars we run an genetic algorithm.

Each generations G_t create a generations G_t+1 with:
- Selection : Geometries with the best scores of G_t
- Crossing : Crossed geometries of two ofast and robust geometries of G_t
- Mutation : Noisy version of fast and robust geometries of G_t
- Injection : Completly new geometries

**Algorithmic structure of the optimal search method :**
![alt text](https://github.com/oscarfossey/Optmisation-Jansen-Mecanism/blob/main/Images/Architecture.JPG)

## Results

The definition and the tunning of the hyperparameters are in the report.

The best geometry has the next nominal lenght:

**Nominal geometry : [1.43, 3.99, 3.97, 3.56, 3.73, 5.29, 4.36, 3.90, 6.19, 4.71, 4.83, 5.86, 0.78]**

**Visualization of the robustness (path of one of the feet) :**

![alt text](https://github.com/oscarfossey/Optmisation-Jansen-Mecanism/blob/main/Images/feet_path.JPG)

## Used packages

![alt text](https://github.com/oscarfossey/Optmisation-Jansen-Mecanism/blob/main/Images/packages.JPG)

## Credits

The project has been made and lead by Oscar Fossey (ENSAM), Arthur Lejeune (ENSAM) and Clément Eiserloh (ENSAM).

