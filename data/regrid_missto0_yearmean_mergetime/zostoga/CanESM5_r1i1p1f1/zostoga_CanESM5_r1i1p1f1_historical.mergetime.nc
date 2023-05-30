CDF   �   
      time       bnds         6   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       ,CanESM5 (2019): 
aerosol: interactive
atmos: CanAM5 (T63L49 native atmosphere, T63 Linear Gaussian Grid; 128 x 64 longitude/latitude; 49 levels; top level 1 hPa)
atmosChem: specified oxidants for aerosols
land: CLASS3.6/CTEM1.2
landIce: specified ice sheets
ocean: NEMO3.4.1 (ORCA1 tripolar grid, 1 deg with refinement to 1/3 deg within 20 degrees of the equator; 361 x 290 longitude/latitude; 45 vertical levels; top grid cell 0-6.19 m)
ocnBgchem: Canadian Model of Ocean Carbon (CMOC); NPZD ecosystem with OMIP prescribed carbonate chemistry
seaIce: LIM2   institution       wCanadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada, Victoria, BC V8P 5C2, Canada    CCCma_model_hash      (3dedf95315d603326fde4f5340dc0519d80d10c0   CCCma_parent_runid        
rc3-pictrl     CCCma_pycmor_hash         (33c30511acc319a98240633965a04ca99c26427e   CCCma_runid       rc3.1-his01    YMDH_branch_time_in_child         1850:01:01:00      YMDH_branch_time_in_parent        5201:01:01:00      activity_id       CMIP   branch_method         Spin-up documentation      branch_time_in_child                 branch_time_in_parent         A2��       contact       %ec.cccma.info-info.ccmac.ec@canada.ca      creation_date         2019-05-01T03:09:01Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.historical.none.r1i1p1f1    grid      �ORCA1 tripolar grid, 1 deg with refinement to 1/3 deg within 20 degrees of the equator; 361 x 290 longitude/latitude; 45 vertical levels; top grid cell 0-6.19 m   
grid_label        gn     history      Tue May 30 16:59:24 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Omon.zostoga.gn.v20190429/zostoga_Omon_CanESM5_historical_r1i1p1f1_gn_185001-201412.1d.yearmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_mergetime/zostoga/CanESM5_r1i1p1f1/zostoga_CanESM5_r1i1p1f1_historical.mergetime.nc
Thu Apr 07 22:52:01 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Omon.zostoga.gn.v20190429/zostoga_Omon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/CanESM5_r1i1p1f1/CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Omon.zostoga.gn.v20190429/zostoga_Omon_CanESM5_historical_r1i1p1f1_gn_185001-201412.1d.yearmean.nc
2019-05-01T03:09:01Z ;rewrote data to be consistent with CMIP for variable zostoga found in table Omon.;
Output from $runid   initialization_index            institution_id        CCCma      mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      CanESM5    parent_time_units         days since 1850-01-01 0:0:0.0      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      
references        mGeophysical Model Development Special issue on CanESM5 (https://www.geosci-model-dev.net/special_issues.html)      	source_id         CanESM5    source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ECreation Date:(20 February 2019) MD5:374fbe5a2bcca535c40f7f23da271e49      title         !CanESM5 output prepared for CMIP6      variable_id       zostoga    variant_label         r1i1p1f1   version       	v20190429      license      �CMIP6 model data produced by The Government of Canada (Canadian Centre for Climate Modelling and Analysis, Environment and Climate Change Canada) is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/959ff333-986a-4f84-b985-484ba1587bd5      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               $   	time_bnds                                 ,   zostoga              	   standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       �The reference state is set to the mean value averaged over years 5201-6151 of the PI control with specified CO2., CMIP_table_comment: There is no CMIP6 request for zosga nor zossga.      original_name         zostoga    history       subcnst             <subcnst             <Aq���   Aq��P   Aq�P   <�jAq�6�   Aq�P   Aq��P   <ޮ�Aq���   Aq��P   Aq��P   <���Aq��   Aq��P   Aq�dP   <�jAq���   Aq�dP   Aq��P   <�K@Aq���   Aq��P   Aq�FP   <��jAq�k�   Aq�FP   Aq��P   <��Aq���   Aq��P   Aq�(P   <�Aq�M�   Aq�(P   Aq��P   <���Aq���   Aq��P   Aq�
P   <�T�Aq�/�   Aq�
P   Aq�{P   <��?Aq���   Aq�{P   Aq��P   <�$Aq��   Aq��P   Aq�]P   <���AqĂ�   Aq�]P   Aq��P   <���Aq���   Aq��P   Aq�?P   <�t�Aq�d�   Aq�?P   Aq˰P   <�Aq���   Aq˰P   Aq�!P   <�jAq�F�   Aq�!P   AqВP   <�w?Aqз�   AqВP   Aq�P   <��jAq�(�   Aq�P   Aq�tP   <�C�Aqՙ�   Aq�tP   Aq��P   <�!�Aq�
�   Aq��P   Aq�VP   <�p?Aq�{�   Aq�VP   Aq��P   <��?Aq���   Aq��P   Aq�8P   <�H�Aq�]�   Aq�8P   Aq�P   <�t�Aq���   Aq�P   Aq�P   <�O?Aq�?�   Aq�P   Aq�P   = JJAq��   Aq�P   Aq��P   = F�Aq�!�   Aq��P   Aq�mP   <�̕Aq��   Aq�mP   Aq��P   <�ҕAq��   Aq��P   Aq�OP   <�<Aq�t�   Aq�OP   Aq��P   = !`Aq���   Aq��P   Aq�1P   <�A�Aq�V�   Aq�1P   Aq��P   =�
Aq���   Aq��P   Aq�P   <�Aq�8�   Aq�P   Aq��P   <�%@Aq���   Aq��P   Aq��P   <��Aq��   Aq��P   ArfP   <�u�Ar��   ArfP   Ar�P   <���Ar��   Ar�P   ArHP   <���Arm�   ArHP   Ar�P   <��Ar��   Ar�P   Ar*P   <�7�ArO�   Ar*P   Ar�P   <�F�Ar��   Ar�P   ArP   <ܣ�Ar1�   ArP   Ar}P   <��Ar��   Ar}P   Ar�P   <��jAr�   Ar�P   Ar_P   <���Ar��   Ar_P   Ar�P   <��Ar��   Ar�P   ArAP   <�GArf�   ArAP   Ar�P   <�jAr��   Ar�P   Ar!#P   <꥕Ar!H�   Ar!#P   Ar#�P   <��Ar#��   Ar#�P   Ar&P   <�>�Ar&*�   Ar&P   Ar(vP   <��Ar(��   Ar(vP   Ar*�P   <���Ar+�   Ar*�P   Ar-XP   <�uAr-}�   Ar-XP   Ar/�P   <�	jAr/��   Ar/�P   Ar2:P   <§�Ar2_�   Ar2:P   Ar4�P   <�XAr4��   Ar4�P   Ar7P   <�=jAr7A�   Ar7P   Ar9�P   <�(@Ar9��   Ar9�P   Ar;�P   <Ӌ@Ar<#�   Ar;�P   Ar>oP   <�Q�Ar>��   Ar>oP   Ar@�P   <�/�ArA�   Ar@�P   ArCQP   <�4�ArCv�   ArCQP   ArE�P   <�D�ArE��   ArE�P   ArH3P   <�
@ArHX�   ArH3P   ArJ�P   <ަ@ArJ��   ArJ�P   ArMP   <���ArM:�   ArMP   ArO�P   <�N�ArO��   ArO�P   ArQ�P   <�x@ArR�   ArQ�P   ArThP   <��?ArT��   ArThP   ArV�P   <��ArV��   ArV�P   ArYJP   <�J�ArYo�   ArYJP   Ar[�P   <��Ar[��   Ar[�P   Ar^,P   <�Ar^Q�   Ar^,P   Ar`�P   <�?Ar`��   Ar`�P   ArcP   <��?Arc3�   ArcP   AreP   =
�Are��   AreP   Arg�P   =K�Arh�   Arg�P   ArjaP   =�`Arj��   ArjaP   Arl�P   =�`Arl��   Arl�P   AroCP   =m�Aroh�   AroCP   Arq�P   =��Arq��   Arq�P   Art%P   =�
ArtJ�   Art%P   Arv�P   =�uArv��   Arv�P   AryP   = ��Ary,�   AryP   Ar{xP   ='�Ar{��   Ar{xP   Ar}�P   =�Ar~�   Ar}�P   Ar�ZP   =
Y�Ar��   Ar�ZP   Ar��P   =�uAr���   Ar��P   Ar�<P   =wuAr�a�   Ar�<P   Ar��P   =8�Ar���   Ar��P   Ar�P   =	SJAr�C�   Ar�P   Ar��P   =	�5Ar���   Ar��P   Ar� P   =
�Ar�%�   Ar� P   Ar�qP   =R_Ar���   Ar�qP   Ar��P   =JAr��   Ar��P   Ar�SP   =�JAr�x�   Ar�SP   Ar��P   =U�Ar���   Ar��P   Ar�5P   =uJAr�Z�   Ar�5P   Ar��P   =�
Ar���   Ar��P   Ar�P   =��Ar�<�   Ar�P   Ar��P   =V�Ar���   Ar��P   Ar��P   =��Ar��   Ar��P   Ar�jP   =F�Ar���   Ar�jP   Ar��P   =#JAr� �   Ar��P   Ar�LP   =�uAr�q�   Ar�LP   Ar��P   =(uAr���   Ar��P   Ar�.P   =Ar�S�   Ar�.P   Ar��P   =PJAr���   Ar��P   Ar�P   =��Ar�5�   Ar�P   Ar��P   =��Ar���   Ar��P   Ar��P   =
�5Ar��   Ar��P   Ar�cP   =`�Ar���   Ar�cP   Ar��P   =�5Ar���   Ar��P   Ar�EP   =͵Ar�j�   Ar�EP   ArĶP   =�Ar���   ArĶP   Ar�'P   =� Ar�L�   Ar�'P   ArɘP   =�Arɽ�   ArɘP   Ar�	P   =g�Ar�.�   Ar�	P   Ar�zP   =��ArΟ�   Ar�zP   Ar��P   =�uAr��   Ar��P   Ar�\P   =�ArӁ�   Ar�\P   Ar��P   =�Ar���   Ar��P   Ar�>P   =�uAr�c�   Ar�>P   ArگP   =յAr���   ArگP   Ar� P   =εAr�E�   Ar� P   ArߑP   =D�Ar߶�   ArߑP   Ar�P   =�uAr�'�   Ar�P   Ar�sP   =d�Ar��   Ar�sP   Ar��P   =$��Ar�	�   Ar��P   Ar�UP   ='D�Ar�z�   Ar�UP   Ar��P   =$e5Ar���   Ar��P   Ar�7P   ="5�Ar�\�   Ar�7P   Ar�P   =$P_Ar���   Ar�P   Ar�P   =)7�Ar�>�   Ar�P   Ar��P   =/�
Ar���   Ar��P   Ar��P   =55
Ar� �   Ar��P   Ar�lP   =;��Ar���   Ar�lP   Ar��P   =A@�Ar��   Ar��P   Ar�NP   =D��Ar�s�   Ar�NP   As�P   =:�JAs��   As�P   As0P   =6�
AsU�   As0P   As�P   =<~�As��   As�P   As	P   =C��As	7�   As	P   As�P   =L�As��   As�P   As�P   =S�
As�   As�P   AseP   =X��As��   AseP   As�P   =a��As��   As�P   AsGP   =ecJAsl�   AsGP   As�P   =i�
As��   As�P   As)P   =qw�AsN�   As)P   As�P   =v��As��   As�P   AsP   =�As0�   AsP   As!|P   =�k�As!��   As!|P   As#�P   =��eAs$�   As#�P   As&^P   =��As&��   As&^P   As(�P   =��EAs(��   As(�P   As+@P   =�~PAs+e�   As+@P   As-�P   =���As-��   As-�P   As0"P   =�|�As0G�   As0"P   As2�P   =��0As2��   As2�P   As5P   =�1�As5)�   As5P   As7uP   =��P