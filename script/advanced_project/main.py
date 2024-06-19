from pre_processing import advanced_pre_process
def main():
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis")
    review_df = advanced_pre_process()
    


if __name__ == '__main__':
    main()