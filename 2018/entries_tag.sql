SELECT work.id, tag.name as 'tag', project as 'project', project_name as 'name', details,
  date(started) as 'started', time(started) as 'hour', date(stopped) as 'stopped',
  strftime('%s',stopped)-strftime('%s', started) as lenght
  FROM work
  INNER JOIN work_tag ON work.id=work_id
  INNER JOIN tag ON tag.id=work_tag.tag_id
  WHERE date(started) >= '2018-01-01' and date(stopped) < '2019-01-01'
  ORDER BY work.id ASC