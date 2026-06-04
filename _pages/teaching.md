---
layout: page
permalink: /teaching/
title: teaching
description: Teaching assistantships and course support.
nav: true
nav_order: 5
---

I have taught and supported undergraduate and graduate courses in computer science, software engineering, and engineering design.

<!-- markdownlint-disable MD033 -->

<div class="courses">
  {% assign courses_by_year = site.teachings | sort: "year" | reverse | group_by: "year" %}
  {% for year_group in courses_by_year %}
    <h2 class="year">{{ year_group.name }}</h2>
    <div class="course-list">
      {% assign year_courses = year_group.items | sort: "term_order" | reverse %}
      {% for course in year_courses %}
        <div class="course-item">
          <h3 class="course-title">
            <a href="{{ course.url | relative_url }}">{{ course.title }}</a>
          </h3>

          <div class="course-meta">
            {% if course.term %}
              <span class="course-term">{{ course.term }}</span>
            {% endif %}

            {% if course.instructor %}
              <span class="course-instructor">{{ course.instructor }}</span>
            {% endif %}
          </div>

          {% if course.description %}
            <div class="course-description">
              {{ course.description | markdownify }}
            </div>
          {% endif %}
        </div>
      {% endfor %}
    </div>

{% endfor %}

</div>

<!-- markdownlint-enable MD033 -->
