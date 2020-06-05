import ad_execute


jobs = {
    'train': {'description': 'this is for test',
              'dates': '2020-06-05 18:04:00',
              'groups': 'event_type_and_category',
              'time_indicator': 'date',
              'feature': 'time_diff',
              'days': 'once'},
    'prediction': {'description': None,
                   'dates': '2020-06-05 01:00:00',
                   'groups': 'event_type_and_category',
                   'time_indicator': 'date',
                   'feature': 'time_diff',
                   'days': 'daily'},
    'parameter_tuning': {'description': None,
                         'dates': '2020-06-05 01:00:00',
                         'groups': 'event_type_and_category',
                         'time_indicator': 'date',
                         'feature': 'time_diff',
                         'days': 'thursdays'}
}


if __name__ == '__main__':
    ad = ad_execute.AnomalyDetection(path='/Users/mac/Desktop/test_ad/test_4', environment='local')
    ad.init()
    ad.run_platform()
    # you must put google connection .json file into the data folder where you create with given path
    ad.create_data_source(data_source_type='googlebigquery',
                          data_query_path="""
                                                SELECT
                                                  fullVisitorId,
                                                  TIMESTAMP_SECONDS(visitStartTime) as date, 
                                                  CASE WHEN type = 'PAGE' THEN 'PAGE'
                                                  ELSE eventAction END as event_type_and_category,
                                                  MIN(time) / 1000 as time_diff
                                                FROM (SELECT 
                                                        geoNetwork.city as city,
                                                        device.browser browser,
                                                        device.deviceCategory deviceCategory,
                                                        visitStartTime,
                                                        hits.eventInfo.eventAction eventAction, 
                                                        hits.eventInfo.eventCategory eventCategory, 
                                                        hits.type type, 
                                                        hits.page.pageTitle pageTitle, 
                                                        hits.time time,
                                                        fullVisitorId as fullVisitorId
                                                      FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
                                                           UNNEST(hits) as hits
                                                     ) as a
                                                WHERE pageTitle != 'Home'
                                                GROUP BY    
                                                            visitStartTime,
                                                            deviceCategory,
                                                            browser,
                                                            city,
                                                            eventAction, 
                                                            eventCategory, 
                                                            type, 
                                                            pageTitle,
                                                            eventCategory,
                                                            fullVisitorId
                                                ORDER BY fullVisitorId, visitStartTime 
                          """,
                          db='flash-clover-268917-c90dc06757de.json',
                          host=None,
                          port=None,
                          user=None,
                          pw=None)

    ad.create_jobs(jobs=jobs)
    ad.manage_train()
