def DT_models(df, name, force=False):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    #df = (pd.read_csv('data/_advanced_project/bow_embeddings.csv'))

    #use DT to predict the sentiment of the reviews
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    print("Training the model...")
    X = df.drop(columns=['sentiment', 'parent_asin'])
    y = df['sentiment'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("evaluating the model...")
    print("Model score: ", classification_report(y_test, model.predict(X_test)))
    print("Model confusion matrix: ", confusion_matrix(y_test, model.predict(X_test)))

    #save the model
    import joblib, os
    os.makedirs('models/_advanced_project', exist_ok=True)
    joblib.dump(model, 'models/_advanced_project/' + name + 'sentiment_model_classImbalance.pkl')
    print("Model saved successfully!")

    #reduce the class imbalance, randomly choose 8400 positive samples
    positive_samples = df[df['sentiment'] == 1].sample(8400)
    negative_samples = df[df['sentiment'] == -1]
    neutral_samples = df[df['sentiment'] == 0]

    balanced_df = pd.concat([positive_samples, negative_samples, neutral_samples])
    print(balanced_df['sentiment'].value_counts())

    #use DT to predict the sentiment of the reviews
    print("Training the model...")
    X = balanced_df.drop(columns=['sentiment', 'parent_asin'])
    y = balanced_df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("evaluating the model...")
    print("Model score: \n", classification_report(y_test, model.predict(X_test)))
    print("Model confusion matrix: \n", confusion_matrix(y_test, model.predict(X_test)))
    #save the model
    import joblib
    os.makedirs('_models/_advanced_project', exist_ok=True)
    joblib.dump(model, '_models/_advanced_project/' + name + 'sentiment_model_classBalance.pkl')