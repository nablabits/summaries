SELECT work.id as 'id', project as 'project', project_name as 'name', 
  activity as 'a_id', activity_name as 'activity',
  date(started) as 'started', time(started) as 'hour', 
  strftime('%s',stopped)-strftime('%s', started) as lenght, details
  FROM work
  WHERE date(started) >= '2018-01-01' and date(stopped) < '2019-01-01'
  ORDER BY datetime(started) ASC
