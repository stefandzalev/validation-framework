 Data quality assessment is one of the most fundamental operations executed during data integration. Data validity, a collection of
validation rules applied to the dataset’s attributes, is part of a specific set
of characteristics, which define the overall data quality. The validation
rules provided by domain experts must be known during the data validation checks. In practice, this is not always the case. Domain experts may
not be available, or their number is insufficient, and the project timeline
may be strict, resulting in unknown data validity rules.
In this paper we present a low code framework for outlier and anomaly
detection as an alternative to traditional data validation rules. The framework comprises statistical and machine learning methods that recognize
outliers and label them as invalid data. We evaluated the accuracy and
scalability of four methods using the TPC-DI benchmark tests, and the
results indicate a high level of accuracy and correctness when the appropriate method is employed for a specific distribution. The framework
is exceptionally effective for attributes with skewed and normal distributions for local and global outliers. Additionally, for different scaling
factors, we observed that the ratio of validation time to d
